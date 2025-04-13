import numpy as np
from cython.parallel import prange
from qtradex.indicators.cache_decorator import cache, float_period

cimport numpy as cnp

# Define the data type for NumPy arrays
ctypedef cnp.float64_t DTYPE_t
@cache
@float_period(3,4,5,6)
def ichimoku(cnp.ndarray[DTYPE_t, ndim=1] high, 
                       cnp.ndarray[DTYPE_t, ndim=1] low, 
                       cnp.ndarray[DTYPE_t, ndim=1] close, 
                       int tenkan_period=9, 
                       int kijun_period=26, 
                       int senkou_b_period=52, 
                       int senkou_span=26):
    """
    Calculate the Ichimoku Kinko Hyo indicator.

    Parameters                                                               :
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - close: A NumPy array of shape (n,) containing close prices.
    - tenkan_period: The period for the Tenkan-sen (default is 9).
    - kijun_period: The period for the Kijun-sen (default is 26).
    - senkou_b_period: The period for the Senkou Span B (default is 52).
    - senkou_span: The number of periods to plot Senkou Span A and B into the future (default is 26).

    Returns:
    - A dictionary containing the calculated Ichimoku components:
      - 'tenkan_sen': Tenkan-sen values
      - 'kijun_sen': Kijun-sen values
      - 'senkou_span_a': Senkou Span A values
      - 'senkou_span_b': Senkou Span B values
      - 'chikou_span': Chikou Span values
    """
    cdef int n = high.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] tenkan_sen = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] kijun_sen = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] senkou_span_a = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] senkou_span_b = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] chikou_span = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t max_high, min_low

    # Calculate Tenkan-sen and Kijun-sen
    for i in range(n):
        if i >= tenkan_period - 1:
            max_high = high[i - tenkan_period + 1:i + 1].max()
            min_low = low[i - tenkan_period + 1:i + 1].min()
            tenkan_sen[i] = (max_high + min_low) / 2

        if i >= kijun_period - 1:
            max_high = high[i - kijun_period + 1:i + 1].max()
            min_low = low[i - kijun_period + 1:i + 1].min()
            kijun_sen[i] = (max_high + min_low) / 2

    # Calculate Senkou Span A and B
    for i in range(n):
        if i >= senkou_span - 1:
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

        if i >= senkou_b_period - 1:
            max_high = high[i - senkou_b_period + 1:i + 1].max()
            min_low = low[i - senkou_b_period + 1:i + 1].min()
            senkou_span_b[i] = (max_high + min_low) / 2

    # Calculate Chikou Span
    for i in range(n):
        if i >= senkou_span:
            chikou_span[i] = close[i - senkou_span]

    return (
        tenkan_sen,
        kijun_sen,
        senkou_span_a,
        senkou_span_b,
        chikou_span
    )

def heikin_ashi(dict hlocv):
    cdef int n = hlocv['high'].shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] ha_open = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] ha_high = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] ha_low = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] ha_close = np.zeros(n, dtype=np.float64)

    # Initialize the first Heikin-Ashi candle
    ha_open[0] = (hlocv['open'][0] + hlocv['close'][0]) / 2
    ha_close[0] = (hlocv['open'][0] + hlocv['high'][0] + hlocv['low'][0] + hlocv['close'][0]) / 4
    ha_high[0] = hlocv['high'][0]
    ha_low[0] = hlocv['low'][0]

    # Calculate Heikin-Ashi candles
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_close[i] = (hlocv['open'][i] + hlocv['high'][i] + hlocv['low'][i] + hlocv['close'][i]) / 4
        ha_high[i] = hlocv['high'][i] if hlocv['high'][i] > ha_open[i] and hlocv['high'][i] > ha_close[i] else max(ha_open[i], ha_close[i])
        ha_low[i] = hlocv['low'][i] if hlocv['low'][i] < ha_open[i] and hlocv['low'][i] < ha_close[i] else min(ha_open[i], ha_close[i])

    return {
        'ha_open': ha_open,
        'ha_high': ha_high,
        'ha_low': ha_low,
        'ha_close': ha_close,
        'ha_volume': hlocv["volume"],
    }


@cache
@float_period(2)
def vortex(cnp.ndarray[DTYPE_t, ndim=1] high, 
                     cnp.ndarray[DTYPE_t, ndim=1] low, 
                     int period=14):
    """
    Calculate the Vortex Indicator (VI).

    Parameters:
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - period: The period for calculating the Vortex Indicator (default is 14).

    Returns:
    - A dictionary containing the calculated Vortex Indicator components:
      - 'vortex_plus': Positive Vortex Indicator (VI+)
      - 'vortex_minus': Negative Vortex Indicator (VI-)
      - 'vortex': Combined Vortex Indicator (VI)
    """
    cdef int n = high.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] vortex_plus = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vortex_minus = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] vortex = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t tr_plus, tr_minus, tr_sum

    # Calculate Vortex Indicator
    for i in range(1, n):
        tr_plus = high[i] - low[i - 1] if high[i] - low[i - 1] > 0 else 0
        tr_minus = low[i] - high[i - 1] if low[i] - high[i - 1] > 0 else 0

        vortex_plus[i] = vortex_plus[i - 1] + tr_plus
        vortex_minus[i] = vortex_minus[i - 1] + tr_minus

        if i >= period:
            vortex_plus[i] -= vortex_plus[i - period]
            vortex_minus[i] -= vortex_minus[i - period]

        if vortex_plus[i] + vortex_minus[i] > 0:
            vortex[i] = vortex_plus[i] / (vortex_plus[i] + vortex_minus[i])
        else:
            vortex[i] = 0.0

    return (
        vortex_plus,
        vortex_minus,
        vortex
    )


@cache
@float_period(1,2,3,4,5)
def kst(cnp.ndarray[DTYPE_t, ndim=1] close, 
        int roc1_period=10, 
        int roc2_period=15, 
        int roc3_period=20, 
        int roc4_period=30, 
        int kst_smoothing=9):
    """
    Calculate the KST (Know Sure Thing) indicator.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - roc1_period: The period for the first rate of change (default is 10).
    - roc2_period: The period for the second rate of change (default is 15).
    - roc3_period: The period for the third rate of change (default is 20).
    - roc4_period: The period for the fourth rate of change (default is 30).
    - kst_smoothing: The period for smoothing the KST (default is 9).

    Returns:
    - A dictionary containing the calculated KST components:
      - 'kst': The KST values
      - 'kst_signal': The smoothed KST values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] kst = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] kst_signal = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t roc1, roc2, roc3, roc4

    # Calculate Rate of Change (ROC) and KST
    for i in range(n):
        if i >= roc1_period:
            roc1 = (close[i] - close[i - roc1_period]) / close[i - roc1_period] * 100
            kst[i] += roc1 * 1  # Weight of 1 for the first ROC

        if i >= roc2_period:
            roc2 = (close[i] - close[i - roc2_period]) / close[i - roc2_period] * 100
            kst[i] += roc2 * 2  # Weight of 2 for the second ROC

        if i >= roc3_period:
            roc3 = (close[i] - close[i - roc3_period]) / close[i - roc3_period] * 100
            kst[i] += roc3 * 3  # Weight of 3 for the third ROC

        if i >= roc4_period:
            roc4 = (close[i] - close[i - roc4_period]) / close[i - roc4_period] * 100
            kst[i] += roc4 * 4  # Weight of 4 for the fourth ROC

    # Smooth the KST
    for i in range(n):
        if i >= kst_smoothing:
            kst_signal[i] = np.mean(kst[i - kst_smoothing + 1:i + 1])

    return (
        kst,
        kst_signal
    )



@cache
@float_period(1,2)
def frama(cnp.ndarray[DTYPE_t, ndim=1] close, 
          int period=14, 
          int fractal_period=2):
    """
    Calculate the Fractal Adaptive Moving Average (FRAMA).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for calculating the FRAMA (default is 14).
    - fractal_period: The period for calculating the fractal dimension (default is 2).

    Returns:
    - A dictionary containing the calculated FRAMA:
      - 'frama': The FRAMA values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] frama = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] fractal_dim = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t sum_high, sum_low, sum_close, fractal_sum, fractal_count

    # Calculate FRAMA
    for i in range(n):
        if i >= period - 1:
            sum_high = np.sum(close[i - period + 1:i + 1])
            sum_low = np.sum(close[i - period + 1:i + 1])
            sum_close = np.sum(close[i - period + 1:i + 1])
            frama[i] = (sum_high + sum_low + sum_close) / (3 * period)

            # Calculate fractal dimension
            if i >= fractal_period - 1:
                fractal_sum = 0.0
                fractal_count = 0
                for j in range(fractal_period):
                    if i - j >= 0:
                        fractal_sum += close[i - j]
                        fractal_count += 1
                fractal_dim[i] = fractal_sum / fractal_count

            # Adjust FRAMA based on fractal dimension
            if fractal_dim[i] > 0:
                frama[i] += (fractal_dim[i] - frama[i]) * (1 / fractal_period)

    return frama


@cache
def zigzag(cnp.ndarray[DTYPE_t, ndim=1] close, 
           float deviation=5.0):
    """
    Calculate the Zig Zag Indicator.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - deviation: The percentage change required to identify a reversal (default is 5.0).

    Returns:
    - A NumPy array containing the calculated Zig Zag values.
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] zigzag = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return zigzag  # Return early if the input array is empty

    cdef int i
    cdef DTYPE_t last_extreme = close[0]
    cdef DTYPE_t last_direction = 0  # 1 for up, -1 for down
    zigzag[0] = last_extreme  # Set the first value of zigzag

    for i in range(1, n):
        # Calculate the percentage change
        change = (close[i] - last_extreme) / last_extreme * 100

        # Check for upward reversal
        if change > deviation and last_direction != 1:
            zigzag[i] = close[i]
            last_extreme = close[i]
            last_direction = 1

        # Check for downward reversal
        elif change < -deviation and last_direction != -1:
            zigzag[i] = close[i]
            last_extreme = close[i]
            last_direction = -1
        else:
            # Carry forward the last known extreme value
            zigzag[i] = zigzag[i - 1]

    return zigzag


def zigzag(cnp.ndarray[DTYPE_t, ndim=1] close, 
           float deviation=5.0):
    """
    Calculate the Zig Zag Indicator.

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - deviation: The percentage change required to identify a reversal (default is 5.0).

    Returns:
    - A tuple containing:
      - 'zigzag': The Zig Zag values
      - 'interpolated': The interpolated line connecting the Zig Zag points
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] zigzag = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return zigzag, np.full(n, np.nan, dtype=np.float64)  # Return early if the input array is empty

    cdef int i
    cdef DTYPE_t last_extreme = close[0]
    cdef DTYPE_t last_direction = 0  # 1 for up, -1 for down
    zigzag[0] = last_extreme  # Set the first value of zigzag

    for i in range(1, n):
        # Calculate the percentage change
        change = (close[i] - last_extreme) / last_extreme * 100

        # Check for upward reversal
        if change > deviation and last_direction != 1:
            zigzag[i] = close[i]
            last_extreme = close[i]
            last_direction = 1

        # Check for downward reversal
        elif change < -deviation and last_direction != -1:
            zigzag[i] = close[i]
            last_extreme = close[i]
            last_direction = -1
        else:
            # Carry forward the last known extreme value
            zigzag[i] = zigzag[i - 1]

    # Post-process to create the interpolated line
    interpolated = np.full(n, np.nan, dtype=np.float64)
    valid_indices = np.where(~np.isnan(zigzag))[0]

    if valid_indices.size > 0:
        # Fill the interpolated array
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            interpolated[start_idx:end_idx + 1] = np.linspace(zigzag[start_idx], zigzag[end_idx], end_idx - start_idx + 1)

        # Flatline the last point
        last_valid_index = valid_indices[-1]
        interpolated[last_valid_index:] = zigzag[last_valid_index]

    return zigzag, interpolated


@cache
@float_period(3,4)
def stochastic_momentum_index(cnp.ndarray[DTYPE_t, ndim=1] close, 
                               cnp.ndarray[DTYPE_t, ndim=1] high, 
                               cnp.ndarray[DTYPE_t, ndim=1] low, 
                               int k_period=14, 
                               int d_period=3):
    """
    Calculate the Stochastic Momentum Index (SMI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - k_period: The period for the SMI calculation (default is 14).
    - d_period: The period for the signal line (default is 3).

    Returns:
    - A tuple containing the calculated SMI values:
      - 'smi': The SMI values
      - 'smi_signal': The smoothed SMI values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] smi = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] smi_signal = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t highest_high, lowest_low, smi_numerator, smi_denominator

    # Calculate SMI
    for i in range(n):
        if i >= k_period - 1:
            highest_high = high[i - k_period + 1:i + 1].max()
            lowest_low = low[i - k_period + 1:i + 1].min()

            smi_numerator = (close[i] - (highest_high + lowest_low) / 2)
            smi_denominator = (highest_high - lowest_low) / 2

            if smi_denominator != 0:
                smi[i] = (smi_numerator / smi_denominator) * 100
            else:
                smi[i] = 0.0

    # Smooth the SMI
    for i in range(n):
        if i >= d_period - 1:
            smi_signal[i] = np.mean(smi[i - d_period + 1:i + 1])

    return (smi, smi_signal)


@cache
@float_period(1,2)
def adaptive_rsi(cnp.ndarray[DTYPE_t, ndim=1] close, 
                 int period=14, 
                 int adaptive_period=5):
    """
    Calculate the Adaptive Relative Strength Index (Adaptive RSI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for calculating the traditional RSI (default is 14).
    - adaptive_period: The period for adapting the RSI sensitivity (default is 5).

    Returns:
    - A tuple containing the calculated Adaptive RSI values:
      - 'adaptive_rsi': The Adaptive RSI values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] rsi = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] adaptive_rsi = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t gain, loss, avg_gain, avg_loss, rs

    # Calculate the traditional RSI
    for i in range(1, n):
        change = close[i] - close[i - 1]
        gain = max(change, 0)
        loss = -min(change, 0)

        if i < period:
            avg_gain = np.mean(rsi[:i + 1]) if i > 0 else 0
            avg_loss = np.mean(rsi[:i + 1]) if i > 0 else 0
        elif i == period:
            avg_gain = np.mean(rsi[i - period + 1:i + 1])
            avg_loss = np.mean(rsi[i - period + 1:i + 1])
        else:
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    # Calculate the Adaptive RSI
    for i in range(n):
        if i >= adaptive_period - 1:
            adaptive_rsi[i] = np.mean(rsi[i - adaptive_period + 1:i + 1])

    return (adaptive_rsi,)





@cache
@float_period(3,4)
def ravi(cnp.ndarray[DTYPE_t, ndim=1] close, 
         cnp.ndarray[DTYPE_t, ndim=1] high, 
         cnp.ndarray[DTYPE_t, ndim=1] low, 
         int short_period=14, 
         int long_period=30):
    """
    Calculate the Range Action Verification Index (RAVI).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - high: A NumPy array of shape (n,) containing high prices.
    - low: A NumPy array of shape (n,) containing low prices.
    - short_period: The short period for RAVI calculation (default is 14).
    - long_period: The long period for RAVI calculation (default is 30).

    Returns:
    - A tuple containing the calculated RAVI values:
      - 'ravi': The RAVI values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] ravi = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t short_avg_range, long_avg_range

    # Calculate RAVI
    for i in range(n):
        if i >= short_period - 1:
            short_avg_range = np.mean(high[i - short_period + 1:i + 1]) - np.mean(low[i - short_period + 1:i + 1])
        else:
            short_avg_range = 0.0

        if i >= long_period - 1:
            long_avg_range = np.mean(high[i - long_period + 1:i + 1]) - np.mean(low[i - long_period + 1:i + 1])
        else:
            long_avg_range = 0.0

        if long_avg_range != 0:
            ravi[i] = (short_avg_range - long_avg_range) / long_avg_range * 100
        else:
            ravi[i] = 0.0

    return (ravi,)




@cache
@float_period(1)
def aema(cnp.ndarray[DTYPE_t, ndim=1] close, 
         int period=14, 
         float alpha=0.1):
    """
    Calculate the Adaptive Exponential Moving Average (AEMA).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - period: The period for calculating the AEMA (default is 14).
    - alpha: The smoothing factor (default is 0.1).

    Returns:
    - A tuple containing the calculated AEMA values:
      - 'aema': The AEMA values
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] aema = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t volatility, prev_aema

    # Initialize the first AEMA value
    if n > 0:
        aema[0] = close[0]

    # Calculate AEMA
    for i in range(1, n):
        # Calculate volatility as the absolute difference from the previous close
        volatility = abs(close[i] - close[i - 1])

        # Adjust alpha based on volatility
        adjusted_alpha = alpha / (1 + volatility)

        # Calculate AEMA
        aema[i] = (adjusted_alpha * close[i]) + ((1 - adjusted_alpha) * aema[i - 1])

    return (aema,)



@cache
@float_period(1,2,3)
def emacd(cnp.ndarray[DTYPE_t, ndim=1] close, 
          int short_period=12, 
          int long_period=26, 
          int signal_period=9):
    """
    Calculate the Exponential Moving Average Convergence Divergence (EMACD).

    Parameters:
    - close: A NumPy array of shape (n,) containing close prices.
    - short_period: The short period for the EMA (default is 12).
    - long_period: The long period for the EMA (default is 26).
    - signal_period: The period for the signal line (default is 9).

    Returns:
    - A tuple containing the calculated EMACD values:
      - 'emacd': The EMACD values
      - 'signal_line': The signal line values
      - 'histogram': The histogram values (EMACD - signal line)
    """
    cdef int n = close.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] ema_short = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] ema_long = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] emacd = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] signal_line = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[DTYPE_t, ndim=1] histogram = np.zeros(n, dtype=np.float64)

    cdef int i
    cdef DTYPE_t alpha_short, alpha_long

    # Calculate the smoothing factors
    alpha_short = 2.0 / (short_period + 1)
    alpha_long = 2.0 / (long_period + 1)

    # Calculate the short EMA
    for i in range(n):
        if i == 0:
            ema_short[i] = close[i]
        else:
            ema_short[i] = (close[i] * alpha_short) + (ema_short[i - 1] * (1 - alpha_short))

    # Calculate the long EMA
    for i in range(n):
        if i == 0:
            ema_long[i] = close[i]
        else:
            ema_long[i] = (close[i] * alpha_long) + (ema_long[i - 1] * (1 - alpha_long))

    # Calculate EMACD
    for i in range(n):
        emacd[i] = ema_short[i] - ema_long[i]

    # Calculate the signal line (EMA of EMACD)
    for i in range(n):
        if i >= signal_period - 1:
            if i == signal_period - 1:
                signal_line[i] = np.mean(emacd[:signal_period])
            else:
                signal_line[i] = (emacd[i] * (2.0 / (signal_period + 1))) + (signal_line[i - 1] * (1 - (2.0 / (signal_period + 1))))

    # Calculate the histogram
    for i in range(n):
        histogram[i] = emacd[i] - signal_line[i]

    return (emacd, signal_line, histogram)


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
