@cache
@float_period(1,2)
def tsi(cnp.ndarray[DTYPE_t, ndim=1] close, 
        int long_period=25, 
        int short_period=13):
    """
    Calculate the Trend Strength Indicator (TSI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - long_period: The long period for TSI calculation (default is 25).
    - short_period: The short period for TSI calculation (default is 13).

    Returns:
    - A tuple containing the calculated TSI values:
      - 'tsi': The TSI values
    """
    cdef int n = close.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] tsi = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] price_change = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] smoothed_price_change = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] smoothed_abs_price_change = np.zeros(n, dtype=np.float64)

    cdef int i

    # Calculate price changes
    for i in range(1, n):
        price_change[i] = close[i] - close[i - 1]

    # Calculate smoothed price change (double smoothing)
    for i in range(n):
        if i == 0:
            smoothed_price_change[i] = price_change[i]
            smoothed_abs_price_change[i] = abs(price_change[i])
        else:
            if i < short_period:
                smoothed_price_change[i] = (smoothed_price_change[i - 1] * (i - 1) + price_change[i]) / i
                smoothed_abs_price_change[i] = (smoothed_abs_price_change[i - 1] * (i - 1) + abs(price_change[i])) / i
            else:
                smoothed_price_change[i] = (smoothed_price_change[i - 1] * (short_period - 1) + price_change[i]) / short_period
                smoothed_abs_price_change[i] = (smoothed_abs_price_change[i - 1] * (short_period - 1) + abs(price_change[i])) / short_period

    # Calculate TSI
    for i in range(n):
        if i >= long_period:
            tsi[i] = (smoothed_price_change[i] / smoothed_abs_price_change[i]) * 100

    return (tsi,)



@cache
@float_period(3,4)
def calculate_smi(cnp.ndarray[cnp.float64_t, ndim=1] close_prices,
                  cnp.ndarray[cnp.float64_t, ndim=1] high_prices,
                  cnp.ndarray[cnp.float64_t, ndim=1] low_prices,
                  int n, int smooth_n):
    """
    Calculate the Stochastic Momentum Index (SMI) for a given set of price data.

    The Stochastic Momentum Index is a variation of the Stochastic Oscillator 
    that provides a refined view of momentum and overbought/oversold conditions.

    Parameters:
    ----------
    close_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of closing prices.

    high_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of high prices.

    low_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of low prices.

    n : int
        The lookback period for calculating the highest high and lowest low.

    smooth_n : int
        The period for smoothing the SMI values using a simple moving average.

    Returns:
    -------
    cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array containing the smoothed SMI values. The length of the array 
        will be the same as the input price arrays, with the first `smooth_n - 1` 
        values being zero (or NaN) due to insufficient data for smoothing.

    Notes:
    -----
    - The SMI is calculated using the formula:
      SMI = ((C - L_n) - (H_n - C)) / ((H_n - L_n) / 2)
      where C is the current closing price, L_n is the lowest low over the last n periods,
      and H_n is the highest high over the last n periods.
    - The function handles cases where the highest high and lowest low are equal to avoid 
      division by zero by returning a SMI value of 0.0 in such cases.
    """
    cdef int length = close_prices.shape[0]
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] smi_values = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] smoothed_smi = np.zeros(length, dtype=np.float64)

    for i in range(n - 1, length):
        L_n = np.min(low_prices[i - n + 1:i + 1])
        H_n = np.max(high_prices[i - n + 1:i + 1])
        C = close_prices[i]

        if H_n - L_n != 0:
            smi_values[i] = ((C - L_n) - (H_n - C)) / ((H_n - L_n) / 2)
        else:
            smi_values[i] = 0.0

    # Smooth the SMI values
    for i in range(smooth_n - 1, length):
        smoothed_smi[i] = np.mean(smi_values[i - smooth_n + 1:i + 1])

    return smoothed_smi



import numpy as np

cimport numpy as cnp

# Function to calculate the Exponential Moving Average (EMA)
cdef double calculate_ema(cnp.ndarray[cnp.float64_t, ndim=1] prices, int period, int index):
    cdef double alpha = 2.0 / (period + 1)
    cdef double ema = prices[index]  # Start with the current price for the first EMA
    for i in range(index + 1, index + period):
        ema = (prices[i] - ema) * alpha + ema
    return ema

@cache
@float_period(3)
def calculate_elder_ray_index(cnp.ndarray[cnp.float64_t, ndim=1] close_prices,
                               cnp.ndarray[cnp.float64_t, ndim=1] high_prices,
                               cnp.ndarray[cnp.float64_t, ndim=1] low_prices,
                               int ema_period):
    """
    Calculate the Elder Ray Index, which uses bull and bear power to gauge market strength 
    and potential reversals.

    The Elder Ray Index consists of two components:
    - Bull Power: The difference between the highest price of the current period and the 
      exponential moving average (EMA) of the closing prices.
    - Bear Power: The difference between the lowest price of the current period and the 
      EMA of the closing prices.

    Parameters:
    ----------
    close_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of closing prices.

    high_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of high prices.

    low_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of low prices.

    ema_period : int
        The period for calculating the Exponential Moving Average (EMA) of the closing prices.

    Returns:
    -------
    tuple
        A tuple containing two 1D arrays:
        - bull_power : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of Bull Power values.
        - bear_power : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of Bear Power values. 

    Notes:
    -----
    - Bull Power is calculated as:
      Bull Power = High - EMA(Close)
    - Bear Power is calculated as:
      Bear Power = Low - EMA(Close)
    - The Elder Ray Index can help traders identify potential market reversals and gauge 
      the strength of the current trend. Positive Bull Power indicates bullish strength, 
      while negative Bear Power indicates bearish strength.
    - The function assumes that the input arrays are of the same length and contain 
      sufficient data for the specified EMA period.
    """
    cdef int length = close_prices.shape[0]
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bull_power = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bear_power = np.zeros(length, dtype=np.float64)
    cdef double ema

    for i in range(length):
        if i >= ema_period - 1:
            ema = calculate_ema(close_prices, ema_period, i)
            bull_power[i] = high_prices[i] - ema
            bear_power[i] = low_prices[i] - ema

    return bull_power, bear_power






cimport numpy as cnp

# Function to calculate the Exponential Moving Average (EMA)
cdef double calculate_ema(cnp.ndarray[cnp.float64_t, ndim=1] prices, int period, int index):
    cdef double alpha = 2.0 / (period + 1)
    cdef double ema = prices[index]  # Start with the current price for the first EMA
    for i in range(index + 1, index + period):
        ema = (prices[i] - ema) * alpha + ema
    return ema

# Function to calculate the Relative Strength Index (RSI)
cdef double calculate_rsi(cnp.ndarray[cnp.float64_t, ndim=1] prices, int period, int index):
    cdef double gain = 0.0
    cdef double loss = 0.0
    cdef double avg_gain
    cdef double avg_loss
    cdef double rsi

    for i in range(index - period + 1, index + 1):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gain += change
        else:
            loss -= change

    avg_gain = gain / period
    avg_loss = loss / period

    if avg_loss == 0:
        return 100.0  # Avoid division by zero

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@cache
@float_period(1,2,3)
def calculate_tdi(cnp.ndarray[cnp.float64_t, ndim=1] close_prices,
                  int rsi_period,
                  int ema_period,
                  int volatility_period):
    """
    Calculate the Traders Dynamic Index (TDI), which combines the Relative Strength Index (RSI),
    moving averages, and volatility bands to provide a comprehensive view of market conditions.

    Parameters:
    ----------
    close_prices : cnp.ndarray[cnp.float64_t, ndim=1]
        A 1D array of closing prices.

    rsi_period : int
        The period for calculating the Relative Strength Index (RSI).

    ema_period : int
        The period for calculating the Exponential Moving Average (EMA) of the RSI.

    volatility_period : int
        The period for calculating the volatility bands.

    Returns:
    -------
    tuple
        A tuple containing three 1D arrays:
        - rsi_values : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of RSI values.
        - ema_rsi : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of EMA values of the RSI.
        - volatility_upper : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of upper volatility bands.
        - volatility_lower : cnp.ndarray[cnp.float64_t, ndim=1]
            A 1D array of lower volatility bands.

    Notes:
    -----
    - The TDI provides insights into market conditions, including overbought and oversold levels,
      trend strength, and potential reversals.
    - The function assumes that the input array contains sufficient data for the specified periods.
    """
    cdef int length = close_prices.shape[0]
    cdef int i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] rsi_values = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ema_rsi = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] volatility_upper = np.zeros(length, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] volatility_lower = np.zeros(length, dtype=np.float64)

    # Calculate RSI values
    for i in range(rsi_period - 1, length):
        rsi_values[i] = calculate_rsi(close_prices, rsi_period, i)

    # Calculate EMA of RSI values
    for i in range(ema_period - 1, length):
        ema_rsi[i] = calculate_ema(rsi_values, ema_period, i)

    # Calculate volatility bands
    for i in range(volatility_period - 1, length):

        volatility_upper[i] = ema_rsi[i] + (np.std(rsi_values[i - volatility_period + 1:i + 1]) * 2)
        volatility_lower[i] = ema_rsi[i] - (np.std(rsi_values[i - volatility_period + 1:i + 1]) * 2)

    return rsi_values, ema_rsi, volatility_upper, volatility_lower



cimport numpy as cnp

# Define the data type for NumPy arrays
DTYPE_t = np.float64
@cache
@float_period(4)
def super_trend(cnp.ndarray[DTYPE_t, ndim=1] high, 
                cnp.ndarray[DTYPE_t, ndim=1] low, 
                cnp.ndarray[DTYPE_t, ndim=1] close, 
                cnp.ndarray[DTYPE_t, ndim=1] atr, 
                float multiplier=3.0):
    """
    Calculate the Super Trend Indicator.

    Parameters:
    - high: A NumPy array of high prices.
    - low: A NumPy array of low prices.
    - close: A NumPy array of close prices.
    - atr: A NumPy array of ATR values.
    - multiplier: The multiplier for the ATR to set the Super Trend bands (default is 3.0).

    Returns:
    - A NumPy array containing the Super Trend values.
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] super_trend = np.full(n, nan, dtype=np.float64)

    # Initialize the first Super Trend value
    super_trend[0] = close[0]  # Starting point

    for i in range(1, n):
        if np.isnan(atr[i]):
            super_trend[i] = super_trend[i - 1]  # Carry forward the last value if ATR is NaN
            continue

        # Calculate the upper and lower bands
        upper_band = close[i] + (multiplier * atr[i])
        lower_band = close[i] - (multiplier * atr[i])

        # Determine the Super Trend value
        if close[i] > super_trend[i - 1]:
            super_trend[i] = min(upper_band, super_trend[i - 1])
        else:
            super_trend[i] = max(lower_band, super_trend[i - 1])

    return super_trend
