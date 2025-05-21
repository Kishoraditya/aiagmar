"""
Decorators Module

This module provides decorators for common patterns like:
- Timing function execution
- Retrying operations
- Caching results
- Logging function calls
- Rate limiting
- Authentication and authorization
- Validation
- Error handling
"""

import time
import functools
import inspect
import hashlib
import json
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

from apps.utils.logger import get_logger
from apps.utils.helpers import retry_operation
from apps.utils.constants import DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY, DEFAULT_RETRY_BACKOFF

# Set up logger
logger = get_logger(__name__)

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])
T = TypeVar('T')

# -----------------------------------------------------------------------------
# Performance Decorators
# -----------------------------------------------------------------------------

def timed(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to measure and log the execution time of a function.
    
    Args:
        name: Optional name for the timer (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer_name = name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                logger.debug(f"Function '{timer_name}' took {elapsed_time:.4f} seconds to execute")
        return cast(F, wrapper)
    return decorator


def async_timed(name: Optional[str] = None) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to measure and log the execution time of an async function.
    
    Args:
        name: Optional name for the timer (defaults to function name)
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer_name = name or func.__name__
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                logger.debug(f"Async function '{timer_name}' took {elapsed_time:.4f} seconds to execute")
        return cast(AsyncF, wrapper)
    return decorator


def timeout(seconds: float, timeout_error: Type[Exception] = TimeoutError) -> Callable[[F], F]:
    """
    Decorator to limit the execution time of a synchronous function.

    Args:
        seconds: Maximum execution time in seconds.
        timeout_error: The exception to raise when the function times out.

    Returns:
        The decorated function.

    Raises:
        timeout_error: If the function execution exceeds the specified time limit.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import threading
            import time

            result = [None] * 1
            exception = [None] * 1

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                # Cannot directly stop a thread in Python, raise error in main thread
                raise timeout_error(f"Function {func.__name__} timed out after {seconds} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return cast(F, wrapper)

    return decorator


def async_timeout(seconds: float, timeout_error: Type[Exception] = TimeoutError) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to limit the execution time of an asynchronous function.

    Args:
        seconds: Maximum execution time in seconds.
        timeout_error: The exception to raise when the function times out.

    Returns:
        The decorated async function.

    Raises:
        timeout_error: If the function execution exceeds the specified time limit.
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise timeout_error(f"Async function {func.__name__} timed out after {seconds} seconds")

        return cast(AsyncF, wrapper)

    return decorator


def measure_performance(logger: Optional[logging.Logger] = None, formatter: Optional[Callable[[str, float], str]] = None) -> Callable[[F], F]:
    """
    Decorator to measure and log function execution time and resource usage (basic).

    Args:
        logger: Optional logger instance (defaults to module logger).
        formatter: Optional function to format the log message.

    Returns:
        Decorated function.
    """
    def default_formatter(func_name: str, execution_time: float) -> str:
        return f"Function '{func_name}' took {execution_time:.4f} seconds."

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger(func.__module__)
            msg_formatter = formatter or default_formatter

            start_time = time.time()
            # You could add resource usage measurement here (e.g., memory) if needed
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                log.debug(msg_formatter(func.__name__, elapsed_time))

        return cast(F, wrapper)

    return decorator


def async_measure_performance(logger: Optional[logging.Logger] = None, formatter: Optional[Callable[[str, float], str]] = None) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to measure and log async function execution time and resource usage (basic).

    Args:
        logger: Optional logger instance (defaults to module logger).
        formatter: Optional function to format the log message.

    Returns:
        Decorated async function.
    """
    def default_formatter(func_name: str, execution_time: float) -> str:
        return f"Async function '{func_name}' took {execution_time:.4f} seconds."

    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger(func.__module__)
            msg_formatter = formatter or default_formatter

            start_time = time.time()
            # You could add resource usage measurement here (e.g., memory) if needed
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                log.debug(msg_formatter(func.__name__, elapsed_time))

        return cast(AsyncF, wrapper)

    return decorator


# -----------------------------------------------------------------------------
# Retry Decorators
# -----------------------------------------------------------------------------

def retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    logger_name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to retry a function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retries
        exceptions: Exception or tuple of exceptions to catch
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (delay * backoff for each retry)
        logger_name: Optional logger name
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_operation(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                exceptions=exceptions,
                delay=delay,
                backoff=backoff,
                logger_name=logger_name or func.__name__
            )
        return cast(F, wrapper)
    return decorator


def async_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    logger_name: Optional[str] = None
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to retry an async function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retries
        exceptions: Exception or tuple of exceptions to catch
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier (delay * backoff for each retry)
        logger_name: Optional logger name
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            current_delay = delay
            log = get_logger(logger_name or func.__name__)
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        log.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    log.warning(f"Retry {retries}/{max_retries} after error: {str(e)}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        
        return cast(AsyncF, wrapper)
    return decorator


# -----------------------------------------------------------------------------
# Caching Decorators
# -----------------------------------------------------------------------------

# Simple in-memory cache
_cache: Dict[str, Tuple[Any, datetime]] = {}
_cache_lock = threading.RLock()

def cached(
    ttl: Optional[int] = 3600,  # Time to live in seconds (1 hour default)
    max_size: int = 1000,       # Maximum cache size
    key_func: Optional[Callable[..., str]] = None  # Custom key function
) -> Callable[[F], F]:
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_func: Custom function to generate cache keys
        
    Returns:
        Decorated function
    """
    def default_key_func(*args: Any, **kwargs: Any) -> str:
        """Default function to generate cache keys."""
        # Convert args and kwargs to a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        
        # Use hash for long keys
        if len(key_str) > 100:
            return hashlib.md5(key_str.encode()).hexdigest()
        return key_str
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = f"{func.__module__}.{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{default_key_func(*args, **kwargs)}"
            
            with _cache_lock:
                # Check if result is in cache and not expired
                if cache_key in _cache:
                    result, expiry = _cache[cache_key]
                    if ttl is None or expiry > datetime.now():
                        return result
                    # Remove expired entry
                    del _cache[cache_key]
                
                # If cache is full, remove oldest entries
                if len(_cache) >= max_size:
                    # Remove 10% of oldest entries
                    entries_to_remove = max(1, max_size // 10)
                    sorted_keys = sorted(_cache.items(), key=lambda x: x[1][1])
                    for i in range(entries_to_remove):
                        if i < len(sorted_keys):
                            del _cache[sorted_keys[i][0]]
                
                # Call function and cache result
                result = func(*args, **kwargs)
                if ttl is None:
                    expiry = datetime.max
                else:
                    expiry = datetime.now() + timedelta(seconds=ttl)
                
                _cache[cache_key] = (result, expiry)
                return result
        
        return cast(F, wrapper)
    
    return decorator


def async_cached(
    ttl: Optional[int] = 3600,  # Time to live in seconds (1 hour default)
    max_size: int = 1000,       # Maximum cache size
    key_func: Optional[Callable[..., str]] = None  # Custom key function
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to cache async function results.
    
    Args:
        ttl: Time to live in seconds (None for no expiration)
        max_size: Maximum cache size
        key_func: Custom function to generate cache keys
        
    Returns:
        Decorated async function
    """
    def default_key_func(*args: Any, **kwargs: Any) -> str:
        """Default function to generate cache keys."""
        # Convert args and kwargs to a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        
        # Use hash for long keys
        if len(key_str) > 100:
            return hashlib.md5(key_str.encode()).hexdigest()
        return key_str
    
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = f"{func.__module__}.{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{default_key_func(*args, **kwargs)}"
            
            with _cache_lock:
                # Check if result is in cache and not expired
                if cache_key in _cache:
                    result, expiry = _cache[cache_key]
                    if ttl is None or expiry > datetime.now():
                        return result
                    # Remove expired entry
                    del _cache[cache_key]
                
                # If cache is full, remove oldest entries
                if len(_cache) >= max_size:
                    # Remove 10% of oldest entries
                    entries_to_remove = max(1, max_size // 10)
                    sorted_keys = sorted(_cache.items(), key=lambda x: x[1][1])
                    for i in range(entries_to_remove):
                        if i < len(sorted_keys):
                            del _cache[sorted_keys[i][0]]
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            
            with _cache_lock:
                if ttl is None:
                    expiry = datetime.max
                else:
                    expiry = datetime.now() + timedelta(seconds=ttl)
                
                _cache[cache_key] = (result, expiry)
                return result
        
        return cast(AsyncF, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Logging Decorators
# -----------------------------------------------------------------------------

def log_call(level: str = 'debug') -> Callable[[F], F]:
    """
    Decorator to log function calls.
    
    Args:
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__name__}"
            log = get_logger(func.__module__)
            
            # Get log method based on level
            log_method = getattr(log, level.lower())
            
            # Format arguments (excluding self/cls for methods)
            arg_str = []
            if args:
                # Skip self/cls for methods
                start_idx = 0
                if inspect.ismethod(func) or (len(args) > 0 and args[0].__class__.__name__ == func.__module__.split('.')[-1]):
                    start_idx = 1
                
                for i, arg in enumerate(args[start_idx:], start_idx):
                    arg_str.append(f"arg{i}={repr(arg)}")
            
            for k, v in kwargs.items():
                arg_str.append(f"{k}={repr(v)}")
            
            # Log function call
            log_method(f"Calling {func_name}({', '.join(arg_str)})")
            
            try:
                result = func(*args, **kwargs)
                
                # Log result (truncate if too long)
                result_str = repr(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                
                log_method(f"{func_name} returned: {result_str}")
                return result
            except Exception as e:
                log.error(f"{func_name} raised: {type(e).__name__}: {str(e)}")
                raise
        
        return cast(F, wrapper)
    
    return decorator


def async_log_call(level: str = 'debug') -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to log async function calls.
    
    Args:
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__name__}"
            log = get_logger(func.__module__)
            
            # Get log method based on level
            log_method = getattr(log, level.lower())
            
            # Format arguments (excluding self/cls for methods)
            arg_str = []
            if args:
                # Skip self/cls for methods
                start_idx = 0
                if inspect.iscoroutinefunction(func) and len(args) > 0 and hasattr(args[0], '__class__'):
                    start_idx = 1
                
                for i, arg in enumerate(args[start_idx:], start_idx):
                    arg_str.append(f"arg{i}={repr(arg)}")
            
            for k, v in kwargs.items():
                arg_str.append(f"{k}={repr(v)}")
            
            # Log function call
            log_method(f"Calling async {func_name}({', '.join(arg_str)})")
            
            try:
                result = await func(*args, **kwargs)
                
                # Log result (truncate if too long)
                result_str = repr(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                
                log_method(f"Async {func_name} returned: {result_str}")
                return result
            except Exception as e:
                log.error(f"Async {func_name} raised: {type(e).__name__}: {str(e)}")
                raise
        
        return cast(AsyncF, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Rate Limiting Decorators
# -----------------------------------------------------------------------------

# Rate limiter state
_rate_limiters: Dict[str, Dict[str, Any]] = {}
_rate_limiter_lock = threading.RLock()

def rate_limit(
    calls: int = 10,           # Number of calls allowed
    period: int = 60,          # Time period in seconds
    key_func: Optional[Callable[..., str]] = None  # Function to generate rate limit keys
) -> Callable[[F], F]:
    """
    Decorator to rate limit function calls.
    
    Args:
        calls: Number of calls allowed in the period
        period: Time period in seconds
        key_func: Function to generate rate limit keys (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate rate limit key
            if key_func:
                rate_key = key_func(*args, **kwargs)
            else:
                rate_key = f"{func.__module__}.{func.__name__}"
            
            with _rate_limiter_lock:
                # Initialize rate limiter if not exists
                if rate_key not in _rate_limiters:
                    _rate_limiters[rate_key] = {
                        'calls': 0,
                        'reset_time': time.time() + period
                    }
                
                limiter = _rate_limiters[rate_key]
                
                # Reset if period has passed
                now = time.time()
                if now > limiter['reset_time']:
                    limiter['calls'] = 0
                    limiter['reset_time'] = now + period
                
                # Check if rate limit exceeded
                if limiter['calls'] >= calls:
                    wait_time = limiter['reset_time'] - now
                    logger.warning(f"Rate limit exceeded for {rate_key}. Try again in {wait_time:.2f} seconds.")
                    raise Exception(f"Rate limit exceeded. Try again in {wait_time:.2f} seconds.")
                
                # Increment call count
                limiter['calls'] += 1
            
            # Call function
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def async_rate_limit(
    calls: int = 10,           # Number of calls allowed in the period
    period: int = 60,          # Time period in seconds
    key_func: Optional[Callable[..., str]] = None  # Function to generate rate limit keys
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to rate limit async function calls.
    
    Args:
        calls: Number of calls allowed in the period
        period: Time period in seconds
        key_func: Function to generate rate limit keys (defaults to function name)
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate rate limit key
            if key_func:
                rate_key = key_func(*args, **kwargs)
            else:
                rate_key = f"{func.__module__}.{func.__name__}"
            
            with _rate_limiter_lock:
                # Initialize rate limiter if not exists
                if rate_key not in _rate_limiters:
                    _rate_limiters[rate_key] = {
                        'calls': 0,
                        'reset_time': time.time() + period
                    }
                
                limiter = _rate_limiters[rate_key]
                
                # Reset if period has passed
                now = time.time()
                if now > limiter['reset_time']:
                    limiter['calls'] = 0
                    limiter['reset_time'] = now + period
                
                # Check if rate limit exceeded
                if limiter['calls'] >= calls:
                    wait_time = limiter['reset_time'] - now
                    logger.warning(f"Rate limit exceeded for {rate_key}. Try again in {wait_time:.2f} seconds.")
                    raise Exception(f"Rate limit exceeded. Try again in {wait_time:.2f} seconds.")
                
                # Increment call count
                limiter['calls'] += 1
            
            # Call function
            return await func(*args, **kwargs)
        
        return cast(AsyncF, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Validation Decorators
# -----------------------------------------------------------------------------

def validate_args(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Decorator to validate function arguments.
    
    Args:
        **validators: Dict mapping argument names to validator functions
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    if not validator(arg_value):
                        raise ValueError(f"Invalid value for argument '{arg_name}': {arg_value}")
            
            # Call function
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


def validate_return(validator: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Decorator to validate function return value.
    
    Args:
        validator: Validator function for return value
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if not validator(result):
                raise ValueError(f"Invalid return value: {result}")
            return result
        
        return cast(F, wrapper)
    
    return decorator


def async_validate_args(**validators: Callable[[Any], bool]) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to validate async function arguments.
    
    Args:
        **validators: Dict mapping argument names to validator functions
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            
            # Validate arguments
            for arg_name, validator in validators.items():
                if arg_name in bound_args.arguments:
                    arg_value = bound_args.arguments[arg_name]
                    if not validator(arg_value):
                        raise ValueError(f"Invalid value for argument '{arg_name}': {arg_value}")
            
            # Call function
            return await func(*args, **kwargs)
        
        return cast(AsyncF, wrapper)
    
    return decorator


def async_validate_return(validator: Callable[[Any], bool]) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to validate async function return value.
    
    Args:
        validator: Validator function for return value
        
    Returns:
        Decorated async function
    """
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = await func(*args, **kwargs)
            if not validator(result):
                raise ValueError(f"Invalid return value: {result}")
            return result
        
        return cast(AsyncF, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Error Handling Decorators
# -----------------------------------------------------------------------------

def handle_exceptions(
    handler: Callable[[Exception], Any],
    *exceptions: Type[Exception]
) -> Callable[[F], F]:
    """
    Decorator to handle exceptions.
    
    Args:
        handler: Function to handle exceptions
        *exceptions: Exception types to catch (defaults to Exception)
        
    Returns:
        Decorated function
    """
    if not exceptions:
        exceptions = (Exception,)
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                return handler(e)
        
        return cast(F, wrapper)
    
    return decorator


def async_handle_exceptions(
    handler: Callable[[Exception], Any],
    *exceptions: Type[Exception]
) -> Callable[[AsyncF], AsyncF]:
    """
    Decorator to handle exceptions in async functions.
    
    Args:
        handler: Function to handle exceptions
        *exceptions: Exception types to catch (defaults to Exception)
        
    Returns:
        Decorated async function
    """
    if not exceptions:
        exceptions = (Exception,)
    
    def decorator(func: AsyncF) -> AsyncF:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                return handler(e)
        
        return cast(AsyncF, wrapper)
    
    return decorator


# -----------------------------------------------------------------------------
# Miscellaneous Decorators
# -----------------------------------------------------------------------------

def deprecated(reason: str) -> Callable[[F], F]:
    """
    Decorator to mark functions as deprecated.
    
    Args:
        reason: Reason for deprecation
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.warning(f"Call to deprecated function {func.__name__}: {reason}")
            return func(*args, **kwargs)
        
        # Add deprecation notice to docstring
        if wrapper.__doc__:
            wrapper.__doc__ = f"DEPRECATED: {reason}\n\n{wrapper.__doc__}"
        else:
            wrapper.__doc__ = f"DEPRECATED: {reason}"
        
        return cast(F, wrapper)
    
    return decorator


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to create a singleton class.
    
    Args:
        cls: Class to make singleton
        
    Returns:
        Singleton class
    """
    instances: Dict[Type, Any] = {}
    
    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return cast(Type[T], get_instance)


def memoize(func: F) -> F:
    """
    Decorator to memoize function results (cache with no expiration).
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache: Dict[str, Any] = {}
    
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create a cache key from the function arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = ":".join(key_parts)
        
        # Use hash for long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()
        
        # Check cache
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    return cast(F, wrapper)


def async_memoize(func: AsyncF) -> AsyncF:
    """
    Decorator to memoize async function results (cache with no expiration).
    
    Args:
        func: Async function to memoize
        
    Returns:
        Memoized async function
    """
    cache: Dict[str, Any] = {}
    
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create a cache key from the function arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = ":".join(key_parts)
        
        # Use hash for long keys
        if len(key) > 100:
            key = hashlib.md5(key.encode()).hexdigest()
        
        # Check cache
        if key not in cache:
            cache[key] = await func(*args, **kwargs)
        
        return cache[key]
    
    return cast(AsyncF, wrapper)


def synchronized(lock: Optional[threading.Lock] = None) -> Callable[[F], F]:
    """
    Decorator to synchronize function execution with a lock.
    
    Args:
        lock: Lock to use (creates a new one if not provided)
        
    Returns:
        Synchronized function
    """
    def decorator(func: F) -> F:
        # Create a lock if not provided
        nonlocal lock
        if lock is None:
            lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return func(*args, **kwargs)
        
        return cast(F, wrapper)
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Example usage
    print("Running decorator examples:")

    # Example of timed decorator
    @timed()
    def slow_function(n: int) -> int:
        """Example slow function."""
        time.sleep(n)
        return n * 2

    # Example of retry decorator
    @retry(max_retries=3, exceptions=(ValueError, KeyError))
    def unreliable_function(succeed_after: int) -> str:
        """Example unreliable function that fails a few times."""
        global _call_count
        _call_count += 1
        if _call_count < succeed_after:
            raise ValueError("Not yet!")
        return "Success!"

    # Example of cached decorator
    @cached(ttl=60)
    def expensive_calculation(n: int) -> int:
        """Example expensive calculation."""
        print(f"Calculating {n}...")
        time.sleep(1)
        return n * n

    # Example of log_call decorator
    @log_call(level='info')
    def greet(name: str) -> str:
        """Example function with logging."""
        return f"Hello, {name}!"

    # Example of rate_limit decorator
    @rate_limit(calls=2, period=5)
    def limited_function() -> str:
        """Example rate-limited function."""
        return "Called limited function"

    # Example of validate_args decorator
    @validate_args(n=lambda n: n > 0)
    def positive_only(n: int) -> int:
        """Example function that only accepts positive numbers."""
        return n * 2

    # Example of handle_exceptions decorator
    @handle_exceptions(lambda e: f"Error: {str(e)}")
    def might_fail(x: int) -> int:
        """Example function that might fail."""
        if x == 0:
            raise ValueError("Cannot be zero")
        return 10 // x

    # Example of deprecated decorator
    @deprecated("Use new_function() instead")
    def old_function() -> str:
        """Example deprecated function."""
        return "This function is deprecated"

    # Example of singleton decorator
    @singleton
    class Config:
        """Example singleton class."""
        def __init__(self):
            self.value = 42

    # Example of memoize decorator
    @memoize
    def fibonacci(n: int) -> int:
        """Calculate fibonacci number (expensive recursive calculation)."""
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    # Example of synchronized decorator
    counter = 0

    @synchronized()
    def increment_counter() -> int:
        """Thread-safe counter increment."""
        global counter
        counter += 1
        return counter

    # Example of timeout decorator
    @timeout(seconds=0.1)
    def slow_sync_func():
        time.sleep(0.5)
        return "timed out"

    @async_timeout(seconds=0.1)
    async def slow_async_func():
        await asyncio.sleep(0.5)
        return "timed out async"

    # Example of measure_performance decorator
    @measure_performance()
    def perf_test_func():
        time.sleep(0.05)
        return "performance measured"

    @async_measure_performance()
    async def async_perf_test_func():
        await asyncio.sleep(0.05)
        return "async performance measured"

    # Run examples
    print("\nTimed decorator:")
    result = slow_function(1)
    print(f"Result: {result}")

    print("\nRetry decorator:")
    _call_count = 0
    try:
        result = unreliable_function(3)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nCached decorator:")
    for i in range(3):
        result = expensive_calculation(5)
        print(f"Result: {result}")

    print("\nLog call decorator:")
    result = greet("World")
    print(f"Result: {result}")

    print("\nRate limit decorator:")
    try:
        for i in range(3):
            result = limited_function()
            print(f"Result {i+1}: {result}")
    except Exception as e:
        print(f"Rate limited: {e}")

    print("\nValidate args decorator:")
    try:
        result = positive_only(5)
        print(f"Result: {result}")
        result = positive_only(-5)
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Validation error: {e}")

    print("\nHandle exceptions decorator:")
    for x in [2, 0]:
        result = might_fail(x)
        print(f"Result for {x}: {result}")

    print("\nDeprecated decorator:")
    result = old_function()
    print(f"Result: {result}")

    print("\nSingleton decorator:")
    config1 = Config()
    config2 = Config()
    print(f"Same instance: {config1 is config2}")

    print("\nMemoize decorator:")
    start = time.time()
    result = fibonacci(30)
    end = time.time()
    print(f"Result: {result}, Time: {end - start:.4f} seconds")

    print("\nSynchronized decorator:")
    threads = []
    for _ in range(10):
        t = threading.Thread(target=increment_counter)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Final counter: {counter}")

    print("\nTimeout decorator:")
    try:
        slow_sync_func()
    except TimeoutError as e:
        print(f"Caught expected timeout: {e}")

    print("\nAsync Timeout decorator:")
    try:
        asyncio.run(slow_async_func())
    except TimeoutError as e:
        print(f"Caught expected async timeout: {e}")

    print("\nMeasure performance decorator:")
    perf_test_func()
    asyncio.run(async_perf_test_func())
