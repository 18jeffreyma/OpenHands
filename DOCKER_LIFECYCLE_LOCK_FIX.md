# Docker Lifecycle Lock Race Condition Fix

## Problem

When running swe_perf evaluations with multiple workers, users experienced intermittent "connection refused" errors (error 111) during HTTP requests to Docker containers. This occurred when:

1. Multiple workers run simultaneously, each with a local process + Docker container
2. Local processes communicate with containers via REST requests  
3. When one container shuts down, Docker updates iptables rules
4. These iptables updates can interrupt ongoing HTTP requests to other containers
5. Result: TCP connection refused errors affecting unrelated containers

## Root Cause

The original Docker lifecycle lock used a simple mutex that treated all operations equally. This allowed race conditions where:
- Container shutdown operations could interrupt ongoing HTTP requests
- No coordination between container lifecycle operations and HTTP requests
- iptables rule updates during container shutdown affected concurrent network requests

## Solution

Implemented a **reader-writer lock pattern** in the Docker lifecycle lock:

### Reader Locks (HTTP Requests)
- Multiple HTTP requests can run concurrently
- Represent operations that don't modify Docker/iptables state
- Safe to run simultaneously with other readers

### Writer Locks (Container Operations) 
- Container start/stop operations get exclusive access
- Represent operations that modify Docker/iptables state  
- Must wait for all existing readers to complete
- Block new readers from starting while active

### Key Behaviors
1. **Writer Priority**: When a container operation starts, it signals other operations to wait
2. **Graceful Completion**: Container operations wait for all ongoing HTTP requests to finish
3. **Exclusive Access**: Only one container operation can run at a time
4. **Concurrent Reads**: Multiple HTTP requests can run simultaneously when no writers are active

## Implementation Details

### Updated Files

**`openhands/runtime/impl/docker/docker_lifecycle_lock.py`**
- Refactored to implement proper reader-writer lock pattern
- Added `acquire_reader()` for HTTP requests
- Added `acquire_writer()` for container operations  
- Maintains backward compatibility with legacy `acquire()` method

**`openhands/runtime/impl/docker/docker_runtime.py`**
- Container startup operations use `acquire_writer()`
- Container shutdown operations use `acquire_writer()`
- Increased timeouts for container operations (60-120 seconds)

**`openhands/runtime/impl/action_execution/action_execution_client.py`**
- HTTP requests use `acquire_reader()`
- Increased timeout for HTTP requests (60 seconds)

### Thread Safety
- Uses `threading.RLock()` for intra-process coordination
- Uses `multiprocessing.Lock()` for inter-process coordination
- Proper state management with atomic operations

## Usage Examples

### HTTP Request (Reader)
```python
with docker_lifecycle_lock.acquire_reader(operation="GET /api/endpoint"):
    response = send_http_request(url)
```

### Container Operation (Writer)
```python
with docker_lifecycle_lock.acquire_writer(operation="start_container_xyz"):
    container = docker_client.containers.run(...)
```

### Legacy Compatibility
```python
# Automatically detects operation type
with docker_lifecycle_lock.acquire(operation="start_container_xyz"):  # Writer
    container = docker_client.containers.run(...)
    
with docker_lifecycle_lock.acquire(operation="GET /api/test"):  # Reader
    response = send_request(url)
```

## Testing

Created comprehensive test suite covering:
- Concurrent reader operations
- Writer blocking new readers
- Writer waiting for existing readers
- Race condition prevention scenarios
- Full swe_perf evaluation simulation

All tests validate that container operations wait for ongoing HTTP requests to complete before proceeding.

## Expected Impact

This fix should eliminate the "connection refused" errors in swe_perf evaluations by ensuring:
1. Container shutdowns wait for all ongoing HTTP requests to complete
2. iptables updates don't interrupt active network connections
3. Multiple HTTP requests can still run concurrently for performance
4. Container operations are properly serialized to prevent conflicts

The reader-writer pattern maintains performance for concurrent HTTP requests while providing the necessary coordination for container lifecycle operations.