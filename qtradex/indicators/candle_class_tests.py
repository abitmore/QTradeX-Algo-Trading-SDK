import unittest
import numpy as np
import talib

class TestCandlestickPatterns(unittest.TestCase):

    def compare_ints(self, expected, actual, message):
        self.assertListEqual(expected, actual, message)

    def test_two_crows(self):
        # Sample data for the Two Crows pattern
        highs = [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 215, 245, 235]
        opens = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 110, 240, 230]
        closes = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 200, 220, 150]
        lows = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 105, 215, 145]

        # Create a SimpleSeries instance
        series = SimpleSeries(highs, opens, closes, lows, [], [])

        # Call the two_crows function
        result = two_crows(series)

        # Use TA-Lib to get the expected result
        expected = talib.CDL2CROWS(np.array(opens), np.array(highs), np.array(lows), np.array(closes))

        # Compare the results
        self.compare_ints(expected.tolist(), result, "result = talib.CDL2CROWS(testOpen,testHigh,testLow,testClose)")

    def test_three_black_crows(self):
        # Sample data for the Three Black Crows pattern
        highs = [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 165, 205, 170, 155]
        opens = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 100, 200, 175, 150]
        closes = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 160, 150, 125, 100]
        lows = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 105, 149, 124, 105]

        # Create a SimpleSeries instance
        series = SimpleSeries(highs, opens, closes, lows, [], [])

        # Call the three_black_crows function
        result = three_black_crows(series)

        # Use TA-Lib to get the expected result
        expected = talib.CDL3BLACKCROWS(np.array(opens), np.array(highs), np.array(lows), np.array(closes))

        # Compare the results
        self.compare_ints(expected.tolist(), result, "result = talib.CDL3BLACKCROWS(testOpen,testHigh,testLow,testClose)")

if __name__ == "__main__":
    unittest.main()
