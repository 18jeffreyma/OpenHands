"""Tests for the Docker lifecycle lock functionality."""

import threading
import time
from unittest.mock import patch

import pytest

from openhands.runtime.impl.docker.docker_lifecycle_lock import DockerLifecycleLock


class TestDockerLifecycleLock:
    """Test the Docker lifecycle lock reader-writer functionality."""

    def setup_method(self):
        """Reset the lock state before each test."""
        # Reset class variables to ensure clean state
        DockerLifecycleLock._lock = None
        DockerLifecycleLock._lock_file = None
        DockerLifecycleLock._readers_active = 0
        DockerLifecycleLock._writer_waiting = False

    def test_single_reader_acquire_release(self):
        """Test that a single reader can acquire and release the lock."""
        lock = DockerLifecycleLock()
        
        with lock.acquire_reader(operation="test_read"):
            assert DockerLifecycleLock._readers_active == 1
        
        assert DockerLifecycleLock._readers_active == 0

    def test_multiple_readers_concurrent(self):
        """Test that multiple readers can acquire the lock concurrently."""
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
        assert start_count == 3
        assert len(results) == 6  # 3 starts + 3 ends

    def test_writer_excludes_readers(self):
        """Test that a writer blocks new readers and waits for existing ones."""
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
        assert results == [
            "reader_1_acquired",
            "reader_1_released", 
            "writer_acquired",
            "writer_released"
        ]

    def test_writer_blocks_new_readers(self):
        """Test that when a writer is waiting, new readers are blocked."""
        lock = DockerLifecycleLock()
        results = []
        
        def long_reader_task():
            with lock.acquire_reader(operation="long_reader"):
                results.append("long_reader_acquired")
                time.sleep(0.3)  # Hold lock for longer
                results.append("long_reader_released")
        
        def writer_task():
            time.sleep(0.1)  # Let long reader start first
            with lock.acquire_writer(operation="writer"):
                results.append("writer_acquired")
                time.sleep(0.1)
                results.append("writer_released")
        
        def blocked_reader_task():
            time.sleep(0.2)  # Start after writer is waiting
            with lock.acquire_reader(operation="blocked_reader"):
                results.append("blocked_reader_acquired")
                results.append("blocked_reader_released")
        
        # Start all threads
        threads = [
            threading.Thread(target=long_reader_task),
            threading.Thread(target=writer_task),
            threading.Thread(target=blocked_reader_task)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # The blocked reader should wait until after the writer completes
        expected_order = [
            "long_reader_acquired",
            "long_reader_released",
            "writer_acquired", 
            "writer_released",
            "blocked_reader_acquired",
            "blocked_reader_released"
        ]
        assert results == expected_order

    def test_legacy_acquire_method_container_operations(self):
        """Test that the legacy acquire method treats container operations as writers."""
        lock = DockerLifecycleLock()
        results = []
        
        def reader_task():
            with lock.acquire_reader(operation="http_request"):
                results.append("reader_acquired")
                time.sleep(0.2)
                results.append("reader_released")
        
        def legacy_container_task():
            time.sleep(0.1)  # Let reader start first
            with lock.acquire(operation="start_container_test"):
                results.append("container_acquired")
                results.append("container_released")
        
        reader_thread = threading.Thread(target=reader_task)
        container_thread = threading.Thread(target=legacy_container_task)
        
        reader_thread.start()
        container_thread.start()
        
        reader_thread.join()
        container_thread.join()
        
        # Container operation should wait for reader (treated as writer)
        assert results == [
            "reader_acquired",
            "reader_released",
            "container_acquired", 
            "container_released"
        ]

    def test_legacy_acquire_method_http_requests(self):
        """Test that the legacy acquire method treats HTTP requests as readers."""
        lock = DockerLifecycleLock()
        results = []
        
        def legacy_http_task(request_id):
            with lock.acquire(operation=f"GET /api/endpoint_{request_id}"):
                results.append(f"http_{request_id}_acquired")
                time.sleep(0.1)
                results.append(f"http_{request_id}_released")
        
        # Start multiple HTTP request threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=legacy_http_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All HTTP requests should run concurrently (treated as readers)
        start_count = sum(1 for r in results if 'acquired' in r)
        assert start_count == 3
        assert len(results) == 6

    def test_timeout_handling_reader(self):
        """Test that reader lock acquisition respects timeout."""
        lock = DockerLifecycleLock()
        
        def blocking_writer():
            with lock.acquire_writer(operation="blocking_writer"):
                time.sleep(0.5)  # Block for longer than timeout
        
        # Start blocking writer
        writer_thread = threading.Thread(target=blocking_writer)
        writer_thread.start()
        
        time.sleep(0.1)  # Ensure writer starts first
        
        # Try to acquire reader with short timeout
        with pytest.raises(TimeoutError, match="Failed to acquire reader lock"):
            with lock.acquire_reader(timeout=0.2, operation="timeout_reader"):
                pass
        
        writer_thread.join()

    def test_timeout_handling_writer(self):
        """Test that writer lock acquisition respects timeout."""
        lock = DockerLifecycleLock()
        
        def long_reader():
            with lock.acquire_reader(operation="long_reader"):
                time.sleep(0.5)  # Hold lock longer than timeout
        
        # Start long-running reader
        reader_thread = threading.Thread(target=long_reader)
        reader_thread.start()
        
        time.sleep(0.1)  # Ensure reader starts first
        
        # Try to acquire writer with short timeout
        with pytest.raises(TimeoutError, match="Failed to wait for readers to complete"):
            with lock.acquire_writer(timeout=0.2, operation="timeout_writer"):
                pass
        
        reader_thread.join()

    def test_is_locked_method(self):
        """Test the is_locked method returns correct status."""
        lock = DockerLifecycleLock()
        
        # Initially not locked
        assert not lock.is_locked()
        
        # Should be locked when writer is active
        with lock.acquire_writer(operation="test_writer"):
            assert lock.is_locked()
        
        # Should not be locked after writer releases
        assert not lock.is_locked()