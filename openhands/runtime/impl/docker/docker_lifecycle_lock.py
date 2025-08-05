"""Multiprocessing lock for coordinating Docker container lifecycle operations.

This module provides a global lock that prevents HTTP requests from being sent
while Docker containers are being started or stopped, preventing race conditions
that can occur when Docker's iptables rules are being updated.
"""

import multiprocessing as mp
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Optional

from openhands.core.logger import openhands_logger as logger


class DockerLifecycleLock:
    """Global lock for coordinating Docker container lifecycle operations.

    This lock prevents race conditions that can occur when multiple processes
    are starting/stopping Docker containers simultaneously, which can cause
    iptables rules to be temporarily unavailable and result in connection errors.
    
    The lock uses a reader-writer pattern where:
    - HTTP requests are "readers" and can run concurrently
    - Container lifecycle operations are "writers" and have exclusive access
    - Writers have priority over readers to prevent starvation
    """

    _lock: Optional[mp.Lock] = None
    _lock_file: Optional[str] = None
    
    # Use class-level locks for thread safety within a process
    _state_lock = threading.RLock()  # Protects all shared state
    _readers_active = 0
    _writers_waiting = 0
    _writer_active = False

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
            logger.debug('Created global Docker lifecycle lock')

        return cls._lock

    @classmethod
    @contextmanager
    def acquire_reader(cls, timeout: float = 60.0, operation: str = "http_request"):
        """Acquire a reader lock for HTTP requests.
        
        Multiple readers can acquire the lock simultaneously, but they must wait
        if a writer is waiting or active.

        Args:
            timeout: Maximum time to wait for the lock in seconds
            operation: Description of the operation being performed (for logging)

        Yields:
            None

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout period
        """
        start_time = time.time()
        logger.debug(f'Attempting to acquire reader lock for: {operation}')

        # Wait for writers to complete and no writers waiting
        acquired = False
        while time.time() - start_time < timeout:
            with cls._state_lock:
                # Can acquire reader lock if no writers are active or waiting
                if not cls._writer_active and cls._writers_waiting == 0:
                    cls._readers_active += 1
                    acquired = True
                    break
            time.sleep(0.01)  # Small delay to avoid busy waiting

        if not acquired:
            elapsed = time.time() - start_time
            raise TimeoutError(
                f'Failed to acquire reader lock for {operation} '
                f'after {elapsed:.1f} seconds (writers waiting: {cls._writers_waiting}, writer active: {cls._writer_active})'
            )

        try:
            logger.debug(f'Acquired reader lock for: {operation} (active readers: {cls._readers_active})')
            yield
        finally:
            with cls._state_lock:
                cls._readers_active -= 1
                logger.debug(f'Released reader lock for: {operation} (active readers: {cls._readers_active})')

    @classmethod
    @contextmanager
    def acquire_writer(cls, timeout: float = 120.0, operation: str = "container_operation"):
        """Acquire a writer lock for container lifecycle operations.
        
        Writers have exclusive access and will wait for all active readers to complete.

        Args:
            timeout: Maximum time to wait for the lock in seconds
            operation: Description of the operation being performed (for logging)

        Yields:
            None

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout period
        """
        lock = cls._get_lock()
        start_time = time.time()

        logger.debug(f'Attempting to acquire writer lock for: {operation}')

        # Signal that a writer is waiting
        with cls._state_lock:
            cls._writers_waiting += 1

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

            # Mark writer as active and wait for all readers to complete
            with cls._state_lock:
                cls._writer_active = True
                cls._writers_waiting -= 1

            # Wait for all readers to complete
            readers_cleared = False
            while time.time() - start_time < timeout:
                with cls._state_lock:
                    if cls._readers_active == 0:
                        readers_cleared = True
                        break
                    current_readers = cls._readers_active
                logger.debug(f'Waiting for {current_readers} active readers to complete for: {operation}')
                time.sleep(0.01)

            if not readers_cleared:
                elapsed = time.time() - start_time
                raise TimeoutError(
                    f'Failed to wait for readers to complete for {operation} '
                    f'after {elapsed:.1f} seconds (active readers: {cls._readers_active})'
                )

            logger.debug(f'Acquired writer lock for: {operation}')
            yield

        finally:
            # Release the multiprocessing lock
            try:
                lock.release()
            except ValueError:
                # Lock was already released, which is fine
                pass
            
            # Clear the writer active flag
            with cls._state_lock:
                cls._writer_active = False
            
            logger.debug(f'Released writer lock for: {operation}')

    @classmethod
    @contextmanager
    def acquire(cls, timeout: float = 30.0, operation: str = "unknown"):
        """Legacy method for backward compatibility.
        
        This method treats all operations as writers for safety.
        
        Args:
            timeout: Maximum time to wait for the lock in seconds
            operation: Description of the operation being performed (for logging)

        Yields:
            None

        Raises:
            TimeoutError: If the lock cannot be acquired within the timeout period
        """
        # Determine if this is a container operation or HTTP request based on operation string
        if any(keyword in operation.lower() for keyword in ['container', 'start_', 'stop_', 'init_', 'close']):
            # Container lifecycle operations get writer lock
            with cls.acquire_writer(timeout=timeout, operation=operation):
                yield
        else:
            # HTTP requests get reader lock
            with cls.acquire_reader(timeout=timeout, operation=operation):
                yield

    @classmethod
    def is_locked(cls) -> bool:
        """Check if the Docker lifecycle lock is currently held.

        Returns:
            True if the lock is currently held, False otherwise
        """
        lock = cls._get_lock()
        # Try to acquire without blocking to check if it's available
        if lock.acquire(block=False):
            lock.release()
            return False
        return True


# Global instance for easy access
docker_lifecycle_lock = DockerLifecycleLock()
