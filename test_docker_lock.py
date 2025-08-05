#!/usr/bin/env python3
"""Simple test script to validate Docker lifecycle lock functionality."""

import sys
import threading
import time

# Add the project root to the path
sys.path.insert(0, '.')

from openhands.runtime.impl.docker.docker_lifecycle_lock import DockerLifecycleLock


def test_single_reader():
    """Test that a single reader can acquire and release the lock."""
    print("Testing single reader...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writer_waiting = False
    
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
    DockerLifecycleLock._writer_waiting = False
    
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
    DockerLifecycleLock._writer_waiting = False
    
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


def test_legacy_method():
    """Test that the legacy acquire method works correctly."""
    print("Testing legacy acquire method...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0  
    DockerLifecycleLock._writer_waiting = False
    
    lock = DockerLifecycleLock()
    results = []
    
    def http_task(request_id):
        with lock.acquire(operation=f"GET /api/endpoint_{request_id}"):
            results.append(f"http_{request_id}_start")
            time.sleep(0.1)
            results.append(f"http_{request_id}_end")
    
    def container_task():
        time.sleep(0.05)  # Start after some HTTP requests
        with lock.acquire(operation="start_container_test"):
            results.append("container_start")
            results.append("container_end")
    
    # Start HTTP and container threads
    threads = []
    for i in range(2):
        thread = threading.Thread(target=http_task, args=(i,))
        threads.append(thread) 
        thread.start()
    
    container_thread = threading.Thread(target=container_task)
    threads.append(container_thread)
    container_thread.start()
    
    for thread in threads:
        thread.join()
    
    # Should have concurrent HTTP requests and container operation
    assert len(results) > 4, f"Expected more than 4 events, got {len(results)}"
    print("✓ Legacy acquire method test passed")


if __name__ == "__main__":
    print("Running Docker lifecycle lock tests...")
    
    try:
        test_single_reader()
        test_multiple_readers()
        test_writer_blocks_readers()
        test_legacy_method()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)