#!/usr/bin/env python3
"""Test script to simulate the swe_perf race condition scenario."""

import sys
sys.path.insert(0, '.')

import threading
import time
import random
import importlib.util

# Create mock modules to avoid dependency issues
class MockLogger:
    @staticmethod
    def debug(msg):
        if '--verbose' in sys.argv:
            print(f'DEBUG: {msg}')

# Set up the mock before loading
sys.modules['openhands.core.logger'] = type('MockModule', (), {'openhands_logger': MockLogger})()

# Load the lock module directly 
spec = importlib.util.spec_from_file_location('docker_lifecycle_lock', 'openhands/runtime/impl/docker/docker_lifecycle_lock.py')
lock_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock_module)

DockerLifecycleLock = lock_module.DockerLifecycleLock


def simulate_swe_perf_scenario():
    """Simulate the swe_perf scenario with multiple workers and containers."""
    print("🧪 Simulating swe_perf evaluation scenario with multiple workers...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writers_waiting = 0
    DockerLifecycleLock._writer_active = False
    
    lock = DockerLifecycleLock()
    results = []
    errors = []
    
    def simulate_worker_http_requests(worker_id, container_id, num_requests=5):
        """Simulate a worker making HTTP requests to its container."""
        for i in range(num_requests):
            try:
                with lock.acquire_reader(timeout=10.0, operation=f"worker_{worker_id}_http_request_{i}"):
                    # Simulate HTTP request processing time
                    request_time = random.uniform(0.05, 0.15)
                    time.sleep(request_time)
                    results.append(f"worker_{worker_id}_request_{i}_success")
            except TimeoutError as e:
                errors.append(f"worker_{worker_id}_request_{i}_timeout: {e}")
            except Exception as e:
                errors.append(f"worker_{worker_id}_request_{i}_error: {e}")
    
    def simulate_container_lifecycle(container_id, delay_before_shutdown=0.5):
        """Simulate a container being started and then shut down."""
        try:
            # Container startup
            with lock.acquire_writer(timeout=10.0, operation=f"start_container_{container_id}"):
                time.sleep(0.1)  # Simulate container startup time
                results.append(f"container_{container_id}_started")
            
            # Wait a bit (container is running)
            time.sleep(delay_before_shutdown)
            
            # Container shutdown - this is where the race condition occurs
            with lock.acquire_writer(timeout=10.0, operation=f"stop_container_{container_id}"):
                time.sleep(0.1)  # Simulate container shutdown and iptables updates
                results.append(f"container_{container_id}_stopped")
        
        except TimeoutError as e:
            errors.append(f"container_{container_id}_timeout: {e}")
        except Exception as e:
            errors.append(f"container_{container_id}_error: {e}")
    
    # Simulate the problematic scenario:
    # - Multiple workers (A, B, C) with their containers
    # - Container A shuts down while workers B and C are making requests
    
    threads = []
    
    # Start container lifecycle threads
    for container_id in ['A', 'B', 'C']:
        container_thread = threading.Thread(
            target=simulate_container_lifecycle, 
            args=(container_id, random.uniform(0.3, 0.8))
        )
        threads.append(container_thread)
        container_thread.start()
        time.sleep(0.1)  # Stagger container starts
    
    # Start worker HTTP request threads  
    for worker_id in ['A', 'B', 'C']:
        worker_thread = threading.Thread(
            target=simulate_worker_http_requests,
            args=(worker_id, worker_id, 8)  # 8 requests per worker
        )
        threads.append(worker_thread)
        worker_thread.start()
        time.sleep(0.05)  # Slight stagger
    
    # Wait for all operations to complete
    for thread in threads:
        thread.join(timeout=15.0)  # 15 second timeout
    
    # Analyze results
    total_requests = sum(1 for r in results if 'request' in r and 'success' in r)
    container_starts = sum(1 for r in results if 'started' in r)
    container_stops = sum(1 for r in results if 'stopped' in r)
    
    print(f"📊 Results:")
    print(f"  ✓ Successful HTTP requests: {total_requests}/24 (expected 24)")
    print(f"  ✓ Container starts: {container_starts}/3 (expected 3)")
    print(f"  ✓ Container stops: {container_stops}/3 (expected 3)")
    print(f"  ❌ Errors: {len(errors)}")
    
    if errors:
        print(f"  Error details:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    # Check for success criteria
    success = (
        len(errors) == 0 and
        total_requests == 24 and
        container_starts == 3 and
        container_stops == 3
    )
    
    if success:
        print("🎉 SUCCESS: All operations completed without race conditions!")
    else:
        print("❌ FAILURE: Race conditions or timeouts detected")
    
    return success


def test_race_condition_prevention():
    """Test that the improved lock prevents the specific race condition."""
    print("🧪 Testing race condition prevention...")
    
    # Reset lock state
    DockerLifecycleLock._lock = None
    DockerLifecycleLock._readers_active = 0
    DockerLifecycleLock._writers_waiting = 0
    DockerLifecycleLock._writer_active = False
    
    lock = DockerLifecycleLock()
    events = []
    
    def concurrent_requests():
        """Simulate multiple concurrent HTTP requests."""
        def single_request(request_id):
            with lock.acquire_reader(operation=f"request_{request_id}"):
                events.append(f"request_{request_id}_start")
                time.sleep(0.2)  # Simulate request processing
                events.append(f"request_{request_id}_end")
        
        # Start multiple concurrent requests
        threads = []
        for i in range(3):
            thread = threading.Thread(target=single_request, args=(i,))
            threads.append(thread)
            thread.start()
            time.sleep(0.05)  # Small stagger to ensure they overlap
        
        for thread in threads:
            thread.join()
    
    def container_shutdown():
        """Simulate container shutdown that should wait for requests."""
        time.sleep(0.1)  # Let requests start
        with lock.acquire_writer(operation="container_shutdown"):
            events.append("container_shutdown_start")
            time.sleep(0.1)  # Simulate iptables update
            events.append("container_shutdown_end")
    
    # Start request and shutdown threads
    request_thread = threading.Thread(target=concurrent_requests)
    shutdown_thread = threading.Thread(target=container_shutdown)
    
    request_thread.start()
    shutdown_thread.start()
    
    request_thread.join()
    shutdown_thread.join()
    
    # Verify that container shutdown waited for all requests to complete
    try:
        shutdown_start_index = events.index("container_shutdown_start")
        request_end_indices = [i for i, event in enumerate(events) if event.endswith("_end") and "request_" in event]
        
        # All request ends should come before container shutdown starts
        all_requests_finished_first = all(i < shutdown_start_index for i in request_end_indices)
        
        print(f"📊 Event sequence:")
        for i, event in enumerate(events):
            marker = "  🔒" if "container_shutdown" in event else "  📡"
            print(f"  {i+1:2d}. {marker} {event}")
        
        if all_requests_finished_first and len(request_end_indices) >= 3:
            print("✅ SUCCESS: Container shutdown correctly waited for all requests to complete")
            return True
        else:
            print("❌ FAILURE: Race condition detected - shutdown did not wait for requests")
            return False
    except ValueError:
        print("❌ FAILURE: Container shutdown event not found")
        return False


if __name__ == "__main__":
    print("🚀 Testing improved Docker lifecycle lock for swe_perf race condition fix")
    print("=" * 80)
    
    success_count = 0
    
    # Test 1: Basic race condition prevention
    if test_race_condition_prevention():
        success_count += 1
    
    print("\n" + "=" * 80)
    
    # Test 2: Full swe_perf scenario simulation  
    if simulate_swe_perf_scenario():
        success_count += 1
    
    print("\n" + "=" * 80)
    print(f"📊 Overall Results: {success_count}/2 tests passed")
    
    if success_count == 2:
        print("🎉 All tests passed! The improved lock should prevent connection refused errors.")
    else:
        print("❌ Some tests failed. The lock implementation may need further improvement.")
        sys.exit(1)