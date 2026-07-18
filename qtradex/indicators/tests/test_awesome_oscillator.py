import unittest

import numpy as np

import qtradex.indicators.qi as qi


class TestAwesomeOscillator(unittest.TestCase):
    def test_awesome_oscillator(self):
        high = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.float64)
        low = np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.float64)

        result = qi.awesome_oscillator(high, low, 2, 4)

        # median price is a straight ramp (9..16), so both SMAs are also
        # straight ramps offset by a constant -> the difference is constant.
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        np.testing.assert_allclose(result, expected)

    def test_awesome_oscillator_flat_market(self):
        # a perfectly flat market has zero momentum
        high = np.full(10, 10.0)
        low = np.full(10, 8.0)

        result = qi.awesome_oscillator(high, low, 2, 4)

        np.testing.assert_allclose(result, np.zeros(7))


if __name__ == "__main__":
    unittest.main()
