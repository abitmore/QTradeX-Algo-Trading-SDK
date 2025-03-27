"""
Synthesized HLOCV via Harmonic Brownian Walk
"""

# DISABLE SELECT PYLINT TESTS
# pylint: disable=broad-except
#
# STANDARD MODULES
import math
import random
import time
from typing import Any, Dict, List

# THIRD PARTY MODULES
import numpy as np

# GLOBAL CONSTANTS
VERSION = "klines synthetic"
HARMONICS = 7  # Number of HARMONICS in the harmonic walk
ACCEL = 1.0 / 10**6  # Cyclic ACCELeration factor
STEP = 7.0 / 10**2  # Random walk amplitude (step size)
FREQ = 2.0 / 10**4  # FREQuency of harmonic oscillations
VOLATILITY = 2.0  # Variance in HLOC (High/Low/Close/Open) values
VOLUME_SIZE = 5.0  # Magnitude of synthesized volume
START = 0.00001  # Starting price for the simulated data
DEPTH = 1000  # Number of candles to generate


def synthesize(storage: Dict[str, float], tick: int) -> float:
    """
    Generate the next harmonic step in the Brownian walk to simulate price movement.

    Args:
        storage (dict): A dictionary holding simulation state (e.g., sine, log periodic values).
        tick (int): The current tick or time step in the simulation.

    Returns:
        float: The next simulated price value.
    """
    # Initialize sine wave with acceleration factor
    storage["sine"] = 1 + ACCEL

    # Generate the harmonic series based on frequency and tick
    for harmonic in range(1, HARMONICS):
        storage["sine"] += (ACCEL / harmonic) * math.sin(FREQ * harmonic * tick)

    # Update the log periodic value by applying the harmonic sine series
    storage["log_periodic"] *= math.pow(storage["sine"], tick)

    # Add random volatility to simulate market fluctuations
    storage["log_periodic"] = ((1 - STEP) + 2 * STEP * random.random()) * storage[
        "log_periodic"
    ]

    return storage["log_periodic"]


def create_dataset() -> Dict[str, Any]:
    """
    Create a synthetic dataset of closing prices and Unix timestamps based on the Brownian walk model.

    Returns:
        dict: A dictionary containing 'unix' timestamps and 'close' prices generated via the synthesize function.
    """
    tick = 0
    data = {"unix": [], "close": []}  # Initialize dataset storage
    storage = {"log_periodic": START}  # Initialize state storage for the simulation

    # Calculate start time (one day before current time minus depth for initial offset)
    begin = time.time() - 86400 * (DEPTH + 1)

    # Generate the synthetic dataset with timestamps and closing prices
    for _ in range(DEPTH + 1):
        tick += 1
        data["unix"].append(begin + 86600 * tick)  # Unix timestamp for each tick
        data["close"].append(synthesize(storage, tick))  # Simulated closing price

    return data


def hlocv_data(data: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """
    Create synthetic kline (OHLCV) data using the closing prices.

    Args:
        data (dict): A dictionary containing 'close' prices and 'open' prices for each time tick.

    Returns:
        dict: A dictionary containing 'open', 'close', 'high', 'low', and 'volume' values as numpy arrays.
    """
    # Remove the first and last value to maintain a proper candle structure
    data["unix"].pop(0)
    d_c = data["close"][:]  # Closing prices
    d_o = data["close"][:]  # Opening prices (initially same as close)

    # Clear the lists for open, close, high, low, and volume to store new values
    data["open"], data["close"], data["high"], data["low"], data["volume"] = (
        [],
        [],
        [],
        [],
        [],
    )

    # Remove first and last elements for candle creation consistency
    d_c.pop(0)
    d_o.pop()

    # Generate the high, low, and volume for each synthetic candle
    for idx in range(len(d_c)):
        oc_max = max(d_o[idx], d_c[idx])  # Max of open and close
        oc_min = min(d_o[idx], d_c[idx])  # Min of open and close
        spread = oc_max - oc_min  # Price range (spread)

        # Define volatility for random fluctuations
        oc1 = 1 - VOLATILITY / 100
        oc2 = 1 + VOLATILITY / 100

        # Apply volatility to the open and close prices
        data["open"].append(d_o[idx] * random.uniform(oc1, oc2))
        data["close"].append(d_c[idx] * random.uniform(oc1, oc2))

        # Calculate high and low based on volatility and price spread
        data["high"].append(
            oc_max + random.random() * random.random() * VOLATILITY * spread
        )
        data["low"].append(
            oc_min - random.random() * random.random() * VOLATILITY * spread
        )

        # Ensure high is always greater than or equal to open/close
        data["high"][-1] = max(data["open"][-1], data["close"][-1], data["high"][-1])

        # Ensure low is always less than or equal to open/close
        data["low"][-1] = min(data["open"][-1], data["close"][-1], data["low"][-1])

        # Calculate volume based on price range and volatility
        data["volume"].append(
            1
            / data["close"][-1]
            * (data["high"][-1] - data["low"][-1])
            * 10**VOLUME_SIZE
        )

    # Return numpy arrays for efficient numerical handling
    return {k: np.array(v) for k, v in data.items()}


def klines_synthetic() -> Dict[str, np.ndarray]:
    """
    Generate a new synthetic dataset of kline (OHLCV) data.

    This function combines the dataset creation and kline generation steps to produce
    synthetic market data with realistic price movement and volatility.

    Returns:
        dict: A dictionary containing 'open', 'close', 'high', 'low', and 'volume' data as numpy arrays.
    """
    return hlocv_data(create_dataset())  # Generate and return the full kline dataset
