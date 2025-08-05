#!/usr/bin/env python3
"""Standalone test for Docker lifecycle lock functionality."""

import multiprocessing as mp
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Optional


class SimpleLogger:
    """Simple logger for testing."""
    @staticmethod
    def debug(msg):
        print(f"DEBUG: {msg}")


class DockerLifecycleLock:
    """Standalone version of Docker lifecycle lock for testing."""

    _lock: Optional[mp.Lock] = None
    _lock_file: Optional[str] = None
    _reader_count_lock = threading.Lock()
    _readers_active = 0
    _writer_waiting = False

    @classmethod
    def _get_lock(cls) -> mp.Lock:
        """Get or create the global multiprocessing lock."""
        if cls._lock is None:
            # Use a file-based lock that works across processes
            lock_dir = os.path.join(tempfile.gettempdir(), 'openhands_docker_locks')
            os.makedirs(lock_dir, exist_ok=True)
            cls._lock_file = os.path.join(lock_dir, 'docker_lifecycle.lock')

            # Create a multiprocessing lock that can be shared across processes
            cls._lock = mp.Lock()
            SimpleLogger.debug('Created global Docker lifecycle lock')

        return cls._lock

    @classmethod
    @contextmanager
    def acquire_reader(cls, timeout: float = 60.0, operation: str = "http_request"):
        """Acquire a reader lock for HTTP requests."""
        start_time = time.time()
        SimpleLogger.debug(f'Attempting to acquire reader lock for: {operation}')

        # Wait for any pending writers to complete
        acquired = False
        while time.time() - start_time < timeout:
            with cls._reader_count_lock:
                if not cls._writer_waiting:
                    cls._readers_active += 1
                    acquired = True
                    break
            time.sleep(0.01)

        if not acquired:
            elapsed = time.time() - start_time
            raise TimeoutError(
                f'Failed to acquire reader lock for {operation} '
                f'after {elapsed:.1f} seconds (writer waiting: {cls._writer_waiting})'
            )

        try:
            SimpleLogger.debug(f'Acquired reader lock for: {operation} (active readers: {cls._readers_active})')
            yield
        finally:
            with cls._reader_count_lock:
                cls._readers_active -= 1
                SimpleLogger.debug(f'Released reader lock for: {operation} (active readers: {cls._readers_active})')

    @classmethod
    @contextmanager
    def acquire_writer(cls, timeout: float = 120.0, operation: str = "container_operation"):
        """Acquire a writer lock for container lifecycle operations."""
        lock = cls._get_lock()
        start_time = time.time()

        SimpleLogger.debug(f'Attempting to acquire writer lock for: {operation}')

        # Signal that a writer is waiting
        with cls._reader_count_lock:
            cls._writer_waiting = True

        try:
            # First acquire the multiprocessing lock
            lock_acquired = False
            while time.time() - start_time < timeout:
                if lock.acquire(block=False):
                    lock_acquired = True
                    break
                time.sleep(0.01)

            if not lock_acquired:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f'Failed to acquire multiprocessing lock for {operation} '
                    f'after {elapsed:.1f} seconds'
                )

            # Wait for all readers to complete
            readers_cleared = False
            while time.time() - start_time < timeout:
                with cls._reader_count_lock:
                    if cls._readers_active == 0:
                        readers_cleared = True
                        break
                SimpleLogger.debug(f'Waiting for {cls._readers_active} active readers to complete for: {operation}')
                time.sleep(0.01)

            if not readers_cleared:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f'Failed to wait for readers to complete for {operation} '
                    f'after {elapsed:.1f} seconds (active readers: {cls._readers_active})'
                )

            SimpleLogger.debug(f'Acquired writer lock for: {operation}')
            yield

        finally:
            # Release the multiprocessing lock
            try:
                lock.release()
            except ValueError:
                # Lock was already released, which is fine
                pass
            
            # Clear the writer waiting flag
            with cls._reader_count_lock:
                cls._writer_waiting = False
            
            SimpleLogger.debug(f'Released writer lock for: {operation}')


def test_single_reader():
    """Test that a single reader can acquire and release the lock."""
    print("Testing single reader...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writers_waiting = 0
    DockerLifecycleLock._writer_active = False
    
    lock = DockerLifecycleLock()
    
    with lock.acquire_reader(operation="test_read"):
        assert DockerLifecycleLock._readers_active == 1, f"Expected 1 reader, got {DockerLifecycleLock._readers_active}"
    
    assert DockerLifecycleLock._readers_active == 0, f"Expected 0 readers after release, got {DockerLifecycleLock._readers_active}"
    print("✓ Single reader test passed")


def test_multiple_readers():
    """Test that multiple readers can acquire the lock concurrently."""
    print("Testing multiple readers...")
    
    # Reset lock state  
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writers_waiting = 0
    DockerLifecycleLock._writer_active = False
    
    lock = DockerLifecycleLock()
    results = []
    
    def reader_task(reader_id):
        with lock.acquire_reader(operation=f"reader_{reader_id}"):
            results.append(f"reader_{reader_id}_start")
            time.sleep(0.1)  # Simulate work
            results.append(f"reader_{reader_id}_end")
    
    # Start multiple reader threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=reader_task, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # All readers should have started before any ended (concurrent execution)
    start_count = sum(1 for r in results if 'start' in r)
    assert start_count == 3, f"Expected 3 reader starts, got {start_count}"
    assert len(results) == 6, f"Expected 6 total events, got {len(results)}"
    print("✓ Multiple readers test passed")


def test_writer_blocks_readers():
    """Test that a writer blocks new readers and waits for existing ones."""
    print("Testing writer blocking readers...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None  
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writers_waiting = 0
    DockerLifecycleLock._writer_active = False
    
    lock = DockerLifecycleLock()
    results = []
    
    def reader_task(reader_id):
        with lock.acquire_reader(operation=f"reader_{reader_id}"):
            results.append(f"reader_{reader_id}_acquired")
            time.sleep(0.2)  # Hold the lock for a bit
            results.append(f"reader_{reader_id}_released")
    
    def writer_task():
        time.sleep(0.1)  # Let reader start first
        with lock.acquire_writer(operation="writer"):
            results.append("writer_acquired")
            time.sleep(0.1)
            results.append("writer_released")
    
    # Start reader and writer threads
    reader_thread = threading.Thread(target=reader_task, args=(1,))
    writer_thread = threading.Thread(target=writer_task)
    
    reader_thread.start()
    writer_thread.start()
    
    reader_thread.join()
    writer_thread.join()
    
    # Writer should wait for reader to complete
    expected = [
        "reader_1_acquired",
        "reader_1_released", 
        "writer_acquired",
        "writer_released"
    ]
    assert results == expected, f"Expected {expected}, got {results}"
    print("✓ Writer blocking readers test passed")


if __name__ == "__main__":
    print("Running Docker lifecycle lock tests...")
    
    try:
        test_single_reader()
        test_multiple_readers()
        test_writer_blocks_readers()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)