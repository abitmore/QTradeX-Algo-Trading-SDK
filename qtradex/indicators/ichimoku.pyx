import numpy as np

cimport numpy as cnp

# Define the data type for NumPy arrays
ctypedef np.float64_t DTYPE_t

def ichimoku_indicator(cnp.ndarray[DTYPE_t, ndim=1] high, 
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
    cdef np.ndarray[DTYPE_t, ndim=1] tenkan_sen = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] kijun_sen = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] senkou_span_a = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] senkou_span_b = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] chikou_span = np.zeros(n, dtype=np.float64)

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

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }
