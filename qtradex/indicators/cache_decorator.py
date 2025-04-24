import hashlib
import warnings
from functools import wraps

import cachetools
import numpy as np

from .utilities import float_period as cython_float_period


def make_hashable(*args, **kwargs):
    # Convert args to a hashable format
    hashable_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            hashable_args.append(hashlib.md5(arg).hexdigest())
        elif isinstance(arg, list):
            hashable_args.append(tuple(arg))
        else:
            hashable_args.append(arg)

    # Convert kwargs to a hashable format
    hashable_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            hashable_kwargs[key] = hashlib.md5(value).hexdigest()
        elif isinstance(arg, list):
            hashable_kwargs[key] = tuple(value)
        else:
            hashable_kwargs[key] = value

    return tuple(hashable_args), frozenset(hashable_kwargs.items())


def cache(func):
    # Create a cache with a specified size
    cache = cachetools.LRUCache(maxsize=256)

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = make_hashable(*args, **kwargs)
        # Generate a hash key for the function arguments
        # key = hash(key)

        # Check if the result is in the cache
        if key in cache:
            return cache[key]

        # Call the function and store the result in the cache
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper


def float_period(*periods):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Catch all warnings
                result = cython_float_period(func, args, periods)

                # Check if any warnings were raised
                if w:
                    for warning in w:
                        print(
                            "A warning occurred during float_period, "
                            "check that your period is not less than 1 and "
                            f"that '{func.__name__}' does not create infinities:\n"
                            + str(warning.message)
                        )
            return result
        return wrapper
    return decorator
