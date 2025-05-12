import logging
import math
from typing import List, Protocol, Union


class RangeType(int):
    REAL_BODY = 0
    HIGH_LOW = 1
    SHADOWS = 2


class CandleSetting:
    def __init__(self, range_type: RangeType, avg_period: int, factor: float) -> None:
        self.range_type = range_type
        self.avg_period = avg_period
        self.factor = factor


# Candle settings
setting_body_long = CandleSetting(RangeType.REAL_BODY, 10, 1.0)
setting_body_very_long = CandleSetting(RangeType.REAL_BODY, 10, 3.0)
setting_body_short = CandleSetting(RangeType.REAL_BODY, 10, 1.0)
setting_body_doji = CandleSetting(RangeType.HIGH_LOW, 10, 0.1)
setting_shadow_long = CandleSetting(RangeType.REAL_BODY, 0, 1.0)
setting_shadow_very_long = CandleSetting(RangeType.REAL_BODY, 0, 2.0)
setting_shadow_short = CandleSetting(RangeType.SHADOWS, 10, 1.0)
setting_shadow_very_short = CandleSetting(RangeType.HIGH_LOW, 10, 0.1)
setting_near = CandleSetting(RangeType.HIGH_LOW, 5, 0.2)
setting_far = CandleSetting(RangeType.HIGH_LOW, 5, 0.6)
setting_equal = CandleSetting(RangeType.HIGH_LOW, 5, 0.05)


class Series(Protocol):
    def len(self) -> int:
        ...

    def high(self, i: int) -> float:
        ...

    def open(self, i: int) -> float:
        ...

    def close(self, i: int) -> float:
        ...

    def low(self, i: int) -> float:
        ...


class SimpleSeries(Series):
    def __init__(
        self,
        highs: List[float],
        opens: List[float],
        closes: List[float],
        lows: List[float],
        volumes: List[float],
        rands: List[float],
    ) -> None:
        self.highs = highs
        self.opens = opens
        self.closes = closes
        self.lows = lows
        self.volumes = volumes
        self.rands = rands

    def len(self) -> int:
        return len(self.highs)

    def high(self, i: int) -> float:
        return self.highs[i]

    def open(self, i: int) -> float:
        return self.opens[i]

    def close(self, i: int) -> float:
        return self.closes[i]

    def low(self, i: int) -> float:
        return self.lows[i]


class EnhancedSeries:
    def __init__(self, series: Series) -> None:
        self.series = series

    def average(self, st: CandleSetting, sum_: float, i: int) -> float:
        a = self.range_of(st, i)
        if st.avg_period != 0:
            a = sum_ / float(st.avg_period)
        b = 1.0 if st.range_type != RangeType.SHADOWS else 2.0
        return st.factor * a / b

    def candle_color(self, i: int) -> "CandleColor":
        return (
            CandleColor.WHITE
            if self.series.close(i) >= self.series.open(i)
            else CandleColor.BLACK
        )

    def high_low_range(self, i: int) -> float:
        return self.series.high(i) - self.series.low(i)

    def is_candle_gap_down(self, i1: int, i2: int) -> bool:
        return self.series.high(i1) < self.series.low(i2)

    def is_candle_gap_up(self, i1: int, i2: int) -> bool:
        return self.series.low(i1) > self.series.high(i2)

    def lower_shadow(self, i: int) -> float:
        return min(self.series.close(i), self.series.open(i)) - self.series.low(i)

    def range_of(self, st: CandleSetting, i: int) -> float:
        return st.range_type.range_of(self, i)

    def real_body(self, i: int) -> float:
        return abs(self.series.close(i) - self.series.open(i))

    def real_body_gap_down(self, i2: int, i1: int) -> bool:
        return max(self.series.open(i2), self.series.close(i2)) < min(
            self.series.open(i1), self.series.close(i1)
        )

    def real_body_gap_up(self, i2: int, i1: int) -> bool:
        return min(self.series.open(i2), self.series.close(i2)) > max(
            self.series.open(i1), self.series.close(i1)
        )

    def upper_shadow(self, i: int) -> float:
        return self.series.high(i) - max(self.series.close(i), self.series.open(i))


class CandleColor(int):
    WHITE = 1
    BLACK = -1

    def is_black(self) -> bool:
        return self == CandleColor.BLACK

    def is_white(self) -> bool:
        return self == CandleColor.WHITE


# Example usage:
if __name__ == "__main__":
    # Sample data for testing
    highs = [1.0, 1.5, 1.2, 1.3]
    opens = [0.9, 1.4, 1.1, 1.2]
    closes = [1.0, 1.3, 1.0, 1.1]
    lows = [0.8, 1.2, 0.9, 1.0]
    volumes = [100, 150, 200, 250]
    rands = [0.1, 0.2, 0.3, 0.4]

    simple_series = SimpleSeries(highs, opens, closes, lows, volumes, rands)
    enhanced_series = EnhancedSeries(simple_series)

    # Example of using the enhanced series
    print(
        "Candle Color at index 1:",
        "White" if enhanced_series.candle_color(1).is_white() else "Black",
    )
    print("High-Low Range at index 2:", enhanced_series.high_low_range(2))
    print("Real Body at index 3:", enhanced_series.real_body(3))


def two_crows(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = setting_body_long.avg_period + 2

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input length (%d) want (%d)", es.len(), start_idx)
        return out_integer

    # Calculate the initial period total for long body candles.
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - 2

    for i in range(body_long_trailing_idx, start_idx - 2):
        body_long_period_total += es.range_of(setting_body_long, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 2).is_white()
            and es.real_body(i - 2)  # 1st: white
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and es.candle_color(i - 1).is_black()  # long
            and es.real_body_gap_up(i - 1, i - 2)  # 2nd: black
            and es.candle_color(i).is_black()  # gapping up
            and es.series.open(i) < es.series.open(i - 1)  # 3rd: black
            and es.series.open(i) > es.series.close(i - 1)
            and es.series.close(i) > es.series.open(i - 2)  # opening within 2nd rb
            and es.series.close(i) < es.series.close(i - 2)
        ):  # closing within 1st rb
            out_integer[i] = -100

        # Update the body long period total
        body_long_period_total += es.range_of(setting_body_long, i - 2) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        body_long_trailing_idx += 1

    return out_integer


def three_black_crows(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = setting_shadow_very_short.avg_period + 3

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input length (%d) want (%d)", es.len(), start_idx)
        return out_integer

    # Initialize the shadow very short period total for the last three candles.
    shadow_very_short_period_total = [0.0, 0.0, 0.0]
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period

    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total[2] += es.range_of(
            setting_shadow_very_short, i - 2
        )
        shadow_very_short_period_total[1] += es.range_of(
            setting_shadow_very_short, i - 1
        )
        shadow_very_short_period_total[0] += es.range_of(setting_shadow_very_short, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 3).is_white()
            and es.candle_color(i - 2).is_black()  # 1st: white
            and es.lower_shadow(i - 2)  # 2nd: black
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[2], i - 2
            )
            and es.candle_color(i - 1).is_black()  # very short lower shadow
            and es.lower_shadow(i - 1)  # 3rd: black
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[1], i - 1
            )
            and es.candle_color(i).is_black()  # very short lower shadow
            and es.lower_shadow(i)  # 4th: black
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[0], i
            )
            and es.series.open(i - 1) < es.series.open(i - 2)  # very short lower shadow
            and es.series.open(i - 1) > es.series.close(i - 2)
            and es.series.open(i)  # 2nd black opens within 1st black's rb
            < es.series.open(i - 1)
            and es.series.open(i) > es.series.close(i - 1)
            and es.series.high(i - 3)  # 3rd black opens within 2nd black's rb
            > es.series.close(i - 2)
            and es.series.close(i - 2)  # 1st black closes under prior candle's high
            > es.series.close(i - 1)
            and es.series.close(i - 1) > es.series.close(i)  # three declining
        ):  # three declining
            out_integer[i] = -100

        # Update the shadow very short period total
        for tot_idx in range(2, 0, -1):
            shadow_very_short_period_total[tot_idx] += es.range_of(
                setting_shadow_very_short, i - tot_idx
            ) - es.range_of(
                setting_shadow_very_short, shadow_very_short_trailing_idx - tot_idx
            )
        shadow_very_short_trailing_idx += 1

    return out_integer


def three_inside(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = min(setting_body_short.avg_period, setting_body_long.avg_period) + 2

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input length (%d) want (%d)", es.len(), start_idx)
        return out_integer

    # Initialize period totals for long and short bodies.
    body_long_period_total = 0.0
    body_short_period_total = 0.0
    body_long_trailing_idx = start_idx - 2 - setting_body_long.avg_period
    body_short_trailing_idx = start_idx - 1 - setting_body_short.avg_period

    for i in range(body_long_trailing_idx, start_idx - 2):
        body_long_period_total += es.range_of(setting_body_long, i)
    for i in range(body_short_trailing_idx, start_idx - 1):
        body_short_period_total += es.range_of(setting_body_short, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.real_body(i - 2)
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and es.real_body(i - 1)  # 1st: long
            <= es.average(setting_body_short, body_short_period_total, i - 1)
            and max(es.series.close(i - 1), es.series.open(i - 1))  # 2nd: short
            < max(es.series.close(i - 2), es.series.open(i - 2))
            and min(es.series.close(i - 1), es.series.open(i - 1))  # engulfed by 1st
            > min(es.series.close(i - 2), es.series.open(i - 2))
            and (
                (
                    es.candle_color(i - 2).is_white()
                    and es.candle_color(i).is_black()
                    and es.series.close(i) < es.series.open(i - 2)
                )
                or (  # 3rd: opposite to 1st
                    es.candle_color(i - 2).is_black()
                    and es.candle_color(i).is_white()
                    and es.series.close(i) > es.series.open(i - 2)
                )
            )
        ):  # and closing out
            out_integer[i] = -int(es.candle_color(i - 2)) * 100

        # Update the body long and short period totals
        body_long_period_total += es.range_of(setting_body_long, i - 2) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        body_short_period_total += es.range_of(setting_body_short, i - 1) - es.range_of(
            setting_body_short, body_short_trailing_idx
        )
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1

    return out_integer


def three_line_strike(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    start_idx = setting_near.avg_period + 3

    if start_idx >= es.len():
        logging.warning("Too few input length (%d) want (%d)", es.len(), start_idx)
        return out_integer

    near_trailing_idx = start_idx - setting_near.avg_period
    near_period_total = [0.0] * 4

    for i in range(near_trailing_idx, start_idx):
        near_period_total[3] += es.range_of(setting_near, i - 3)
        near_period_total[2] += es.range_of(setting_near, i - 2)

    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 3) == es.candle_color(i - 2)
            and es.candle_color(i - 2) == es.candle_color(i - 1)
            and es.candle_color(i) == -es.candle_color(i - 1)
            and es.open(i - 2)
            >= min(es.open(i - 3), es.close(i - 3))
            - es.average(setting_near, near_period_total[3], i - 3)
            and es.open(i - 2)
            <= max(es.open(i - 3), es.close(i - 3))
            + es.average(setting_near, near_period_total[3], i - 3)
            and es.open(i - 1)
            >= min(es.open(i - 2), es.close(i - 2))
            - es.average(setting_near, near_period_total[2], i - 2)
            and es.open(i - 1)
            <= max(es.open(i - 2), es.close(i - 2))
            + es.average(setting_near, near_period_total[2], i - 2)
            and (
                (
                    es.candle_color(i - 1) == 1
                    and es.close(i - 1) > es.close(i - 2)
                    and es.close(i - 2) > es.close(i - 3)
                    and es.open(i) > es.close(i - 1)
                    and es.close(i) < es.open(i - 3)
                )
                or (
                    es.candle_color(i - 1) == -1
                    and es.close(i - 1) < es.close(i - 2)
                    and es.close(i - 2) < es.close(i - 3)
                    and es.open(i) < es.close(i - 1)
                    and es.close(i) > es.open(i - 3)
                )
            )
        ):
            out_integer[i] = int(es.candle_color(i - 1)) * 100

        for tot_idx in range(3, 1, -1):
            near_period_total[tot_idx] += es.range_of(
                setting_near, i - tot_idx
            ) - es.range_of(setting_near, near_trailing_idx - tot_idx)
        near_trailing_idx += 1

    return out_integer


def three_outside(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    start_idx = 3

    if start_idx >= es.len():
        logging.warning("Too few input length (%d) want (%d)", es.len(), start_idx)
        return out_integer

    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 1) == 1
            and es.candle_color(i - 2) == -1
            and es.close(i - 1) > es.open(i - 2)
            and es.open(i - 1) < es.close(i - 2)
            and es.close(i) > es.close(i - 1)
        ) or (
            es.candle_color(i - 1) == -1
            and es.candle_color(i - 2) == 1
            and es.open(i - 1) > es.close(i - 2)
            and es.close(i - 1) < es.open(i - 2)
            and es.close(i) < es.close(i - 1)
        ):
            out_integer[i] = int(es.candle_color(i - 1)) * 100

    return out_integer


def three_stars_in_south(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = (
        max(
            max(setting_shadow_very_short.avg_period, setting_shadow_long.avg_period),
            max(setting_body_long.avg_period, setting_body_short.avg_period),
        )
        + 2
    )

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning(f"Too few input len({es.len()}) want({start_idx})")
        return out_integer

    # Initialize period totals
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - setting_body_long.avg_period
    shadow_long_period_total = 0.0
    shadow_long_trailing_idx = start_idx - setting_shadow_long.avg_period
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period
    body_short_period_total = 0.0
    body_short_trailing_idx = start_idx - setting_body_short.avg_period

    shadow_very_short_period_total = [0.0, 0.0]

    # Calculate initial period totals
    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total += es.range_of(setting_body_long, i - 2)
    for i in range(shadow_long_trailing_idx, start_idx):
        shadow_long_period_total += es.range_of(setting_shadow_long, i - 2)
    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total[1] += es.range_of(
            setting_shadow_very_short, i - 1
        )
        shadow_very_short_period_total[0] += es.range_of(setting_shadow_very_short, i)
    for i in range(body_short_trailing_idx, start_idx):
        body_short_period_total += es.range_of(setting_body_short, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 2).is_black()
            and es.candle_color(i - 1).is_black()  # 1st black
            and es.candle_color(i).is_black()  # 2nd black
            and  # 3rd black
            # 1st: long
            es.real_body(i - 2)
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and
            # with long lower shadow
            es.lower_shadow(i - 2)
            > es.average(setting_shadow_long, shadow_long_period_total, i - 2)
            and es.real_body(i - 1) < es.real_body(i - 2)
            and es.open(i - 1) > es.close(i - 2)  # 2nd: smaller candle
            and es.open(i - 1) <= es.high(i - 2)
            and es.low(i - 1) < es.close(i - 2)  # opens higher but within 1st range
            and es.low(i - 1) >= es.low(i - 2)  # trades lower than 1st close
            and  # not lower than 1st low
            # and has a lower shadow
            es.lower_shadow(i - 1)
            > es.average(
                setting_shadow_very_short, shadow_very_short_period_total[1], i - 1
            )
            and
            # 3rd: small marubozu
            es.real_body(i) < es.average(setting_body_short, body_short_period_total, i)
            and es.lower_shadow(i)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[0], i
            )
            and es.upper_shadow(i)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[0], i
            )
            and es.low(i) > es.low(i - 1)
            and es.high(i) < es.high(i - 1)
        ):  # engulfed by prior candle's range
            out_integer[i] = 100

    return out_integer


def three_white_soldiers(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = (
        max(
            max(setting_shadow_very_short.avg_period, setting_body_short.avg_period),
            max(setting_far.avg_period, setting_near.avg_period),
        )
        + 2
    )

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning(f"Too few input len({es.len()}) want({start_idx})")
        return out_integer

    # Initialize period totals
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period
    near_trailing_idx = start_idx - setting_near.avg_period
    far_trailing_idx = start_idx - setting_far.avg_period
    body_short_period_total = 0.0
    body_short_trailing_idx = start_idx - setting_body_short.avg_period

    shadow_very_short_period_total = [0.0, 0.0, 0.0]
    near_period_total = [0.0, 0.0, 0.0]
    far_period_total = [0.0, 0.0, 0.0]

    # Calculate initial period totals
    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total[2] += es.range_of(
            setting_shadow_very_short, i - 2
        )
        shadow_very_short_period_total[1] += es.range_of(
            setting_shadow_very_short, i - 1
        )
        shadow_very_short_period_total[0] += es.range_of(setting_shadow_very_short, i)
    for i in range(near_trailing_idx, start_idx):
        near_period_total[2] += es.range_of(setting_near, i - 2)
        near_period_total[1] += es.range_of(setting_near, i - 1)
    for i in range(far_trailing_idx, start_idx):
        far_period_total[2] += es.range_of(setting_far, i - 2)
        far_period_total[1] += es.range_of(setting_far, i - 1)
    for i in range(body_short_trailing_idx, start_idx):
        body_short_period_total += es.range_of(setting_body_short, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 2).is_white()
            and es.upper_shadow(i - 2)  # 1st white
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[2], i - 2
            )
            and
            # very short upper shadow
            es.candle_color(i - 1).is_white()
            and es.upper_shadow(i - 1)  # 2nd white
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[1], i - 1
            )
            and
            # very short upper shadow
            es.candle_color(i).is_white()
            and es.upper_shadow(i)  # 3rd white
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[0], i
            )
            and
            # very short upper shadow
            es.close(i) > es.close(i - 1)
            and es.close(i - 1) > es.close(i - 2)
            and es.open(i - 1) > es.open(i - 2)  # consecutive higher closes
            and es.open(i - 1)  # 2nd opens within/near 1st real body
            <= es.close(i - 2) + es.average(setting_near, near_period_total[2], i - 2)
            and es.open(i) > es.open(i - 1)
            and es.open(i)  # 3rd opens within/near 2nd real body
            <= es.close(i - 1) + es.average(setting_near, near_period_total[1], i - 1)
            and es.real_body(i - 1)
            > es.real_body(i - 2) - es.average(setting_far, far_period_total[2], i - 2)
            and
            # 2nd not far shorter than 1st
            es.real_body(i)
            > es.real_body(i - 1) - es.average(setting_far, far_period_total[1], i - 1)
            and
            # 3rd not far shorter than 2nd
            es.real_body(i)
            > es.real_body(i - 2) - es.average(setting_far, far_period_total[2], i - 2)
        ):  # not short real body
            out_integer[i] = 100  # Pattern identified

        # Update period totals after pattern recognition
        for tot_idx in range(2, -1, -1):
            shadow_very_short_period_total[tot_idx] += es.range_of(
                setting_shadow_very_short, i - tot_idx
            ) - es.range_of(
                setting_shadow_very_short, shadow_very_short_trailing_idx - tot_idx
            )
        for tot_idx in range(2, 0, -1):
            far_period_total[tot_idx] += es.range_of(
                setting_far, i - tot_idx
            ) - es.range_of(setting_far, far_trailing_idx - tot_idx)
            near_period_total[tot_idx] += es.range_of(
                setting_near, i - tot_idx
            ) - es.range_of(setting_near, near_trailing_idx - tot_idx)
        body_short_period_total += es.range_of(setting_body_short, i) - es.range_of(
            setting_body_short, body_short_trailing_idx
        )

        # Increment trailing indices
        shadow_very_short_trailing_idx += 1
        near_trailing_idx += 1
        far_trailing_idx += 1
        body_short_trailing_idx += 1

    return out_integer


def abandoned_baby(series: Series, penetration: float) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    if penetration == DefaultFloat64:
        penetration = 0.3
    elif penetration < 0.0 or penetration > 3e37:
        logging.warning("Penetration out of range")
        return out_integer

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = (
        max(
            max(setting_body_doji.avg_period, setting_body_long.avg_period),
            setting_body_short.avg_period,
        )
        + 2
    )

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning(f"Too few input len({es.len()}) want({start_idx})")
        return out_integer

    # Initialize period totals
    body_long_period_total = 0.0
    body_doji_period_total = 0.0
    body_short_period_total = 0.0
    body_long_trailing_idx = start_idx - 2 - setting_body_long.avg_period
    body_doji_trailing_idx = start_idx - 1 - setting_body_doji.avg_period
    body_short_trailing_idx = start_idx - setting_body_short.avg_period

    for i in range(body_long_trailing_idx, start_idx - 2):
        body_long_period_total += es.range_of(setting_body_long, i)
    for i in range(body_doji_trailing_idx, start_idx - 1):
        body_doji_period_total += es.range_of(setting_body_doji, i)
    for i in range(body_short_trailing_idx, start_idx):
        body_short_period_total += es.range_of(setting_body_short, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.real_body(i - 2)
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and es.real_body(i - 1)  # 1st: long
            <= es.average(setting_body_doji, body_doji_period_total, i - 1)
            and es.real_body(i)  # 2nd: doji
            > es.average(setting_body_short, body_short_period_total, i)
            and (  # 3rd: longer than short
                (
                    es.candle_color(i - 2) == 1
                    and es.candle_color(i) == -1  # 1st white
                    and es.close(i)  # 3rd black
                    < es.close(i - 2) - es.real_body(i - 2) * penetration
                    and es.is_candle_gap_up(  # 3rd closes well within 1st rb
                        i - 1, i - 2
                    )
                    and es.is_candle_gap_down(  # upside gap between 1st and 2nd
                        i, i - 1
                    )
                )
                or (  # downside gap between 2nd and 3rd
                    es.candle_color(i - 2) == -1
                    and es.candle_color(i) == 1  # 1st black
                    and es.close(i)  # 3rd white
                    > es.close(i - 2) + es.real_body(i - 2) * penetration
                    and es.is_candle_gap_down(  # 3rd closes well within 1st rb
                        i - 1, i - 2
                    )
                    and es.is_candle_gap_up(  # downside gap between 1st and 2nd
                        i, i - 1
                    )
                )
            )
        ):  # upside gap between 2nd and 3rd
            out_integer[i] = (
                int(es.candle_color(i)) * 100
            )  # Set output based on candle color

        # Update period totals after pattern recognition
        body_long_period_total += es.range_of(setting_body_long, i - 2) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        body_doji_period_total += es.range_of(setting_body_doji, i - 1) - es.range_of(
            setting_body_doji, body_doji_trailing_idx
        )
        body_short_period_total += es.range_of(setting_body_short, i) - es.range_of(
            setting_body_short, body_short_trailing_idx
        )

        # Increment trailing indices
        body_long_trailing_idx += 1
        body_doji_trailing_idx += 1
        body_short_trailing_idx += 1

    return out_integer


def advance_block(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = (
        max(
            max(
                max(setting_shadow_long.avg_period, setting_shadow_short.avg_period),
                max(setting_far.avg_period, setting_near.avg_period),
            ),
            setting_body_long.avg_period,
        )
        + 2
    )

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Initialize trailing indices and totals
    shadow_short_trailing_idx = start_idx - setting_shadow_short.avg_period
    shadow_long_trailing_idx = start_idx - setting_shadow_long.avg_period
    near_trailing_idx = start_idx - setting_near.avg_period
    far_trailing_idx = start_idx - setting_far.avg_period
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - setting_body_long.avg_period

    shadow_short_period_total = [0.0, 0.0, 0.0]
    shadow_long_period_total = [0.0, 0.0]
    near_period_total = [0.0, 0.0, 0.0]
    far_period_total = [0.0, 0.0, 0.0]

    # Calculate initial period totals
    for i in range(shadow_short_trailing_idx, start_idx):
        shadow_short_period_total[2] += es.range_of(setting_shadow_short, i - 2)
        shadow_short_period_total[1] += es.range_of(setting_shadow_short, i - 1)
        shadow_short_period_total[0] += es.range_of(setting_shadow_short, i)

    for i in range(shadow_long_trailing_idx, start_idx):
        shadow_long_period_total[1] += es.range_of(setting_shadow_long, i - 1)
        shadow_long_period_total[0] += es.range_of(setting_shadow_long, i)

    for i in range(near_trailing_idx, start_idx):
        near_period_total[2] += es.range_of(setting_near, i - 2)
        near_period_total[1] += es.range_of(setting_near, i - 1)

    for i in range(far_trailing_idx, start_idx):
        far_period_total[2] += es.range_of(setting_far, i - 2)
        far_period_total[1] += es.range_of(setting_far, i - 1)

    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total += es.range_of(setting_body_long, i - 2)

    # Proceed with the calculation for the requested range
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 2) == CandleColor.WHITE
            and es.candle_color(i - 1) == CandleColor.WHITE  # 1st white
            and es.candle_color(i) == CandleColor.WHITE  # 2nd white
            and es.close(i) > es.close(i - 1) > es.close(i - 2)  # 3rd white
            and es.open(i - 1) > es.open(i - 2)  # consecutive higher closes
            and es.open(i - 1)  # 2nd opens within/near 1st real body
            <= es.close(i - 2) + es.average(setting_near, near_period_total[2], i - 2)
            and es.open(i) > es.open(i - 1)
            and es.open(i)  # 3rd opens within/near 2nd real body
            <= es.close(i - 1) + es.average(setting_near, near_period_total[1], i - 1)
            and es.real_body(i - 2)
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and es.upper_shadow(i - 2)  # 1st: long real body
            < es.average(setting_shadow_short, shadow_short_period_total[2], i - 2)
            and (  # 1st: short upper shadow
                (
                    es.real_body(i - 1)
                    < es.real_body(i - 2)
                    - es.average(setting_far, far_period_total[2], i - 2)
                    and es.real_body(i)
                    < es.real_body(i - 1)
                    + es.average(setting_near, near_period_total[1], i - 1)
                )
                or (
                    es.real_body(i)
                    < es.real_body(i - 1)
                    - es.average(setting_far, far_period_total[1], i - 1)
                )
                or (
                    es.real_body(i) < es.real_body(i - 1)
                    and es.real_body(i - 1) < es.real_body(i - 2)
                    and (
                        es.upper_shadow(i)
                        > es.average(
                            setting_shadow_short, shadow_short_period_total[0], i
                        )
                        or es.upper_shadow(i - 1)
                        > es.average(
                            setting_shadow_short, shadow_short_period_total[1], i - 1
                        )
                    )
                )
                or (
                    es.real_body(i) < es.real_body(i - 1)
                    and es.upper_shadow(i)
                    > es.average(setting_shadow_long, shadow_long_period_total[0], i)
                )
            )
        ):
            out_integer[i] = -100

        # Update the period totals for the next iteration
        for tot_idx in range(2, -1, -1):
            shadow_short_period_total[tot_idx] += es.range_of(
                setting_shadow_short, i - tot_idx
            ) - es.range_of(setting_shadow_short, shadow_short_trailing_idx - tot_idx)

        for tot_idx in range(1, -1, -1):
            shadow_long_period_total[tot_idx] += es.range_of(
                setting_shadow_long, i - tot_idx
            ) - es.range_of(setting_shadow_long, shadow_long_trailing_idx - tot_idx)

        for tot_idx in range(2, 0, -1):
            far_period_total[tot_idx] += es.range_of(
                setting_far, i - tot_idx
            ) - es.range_of(setting_far, far_trailing_idx - tot_idx)
            near_period_total[tot_idx] += es.range_of(
                setting_near, i - tot_idx
            ) - es.range_of(setting_near, near_trailing_idx - tot_idx)

        body_long_period_total += es.range_of(setting_body_long, i - 2) - es.range_of(
            setting_body_long, body_long_trailing_idx - 2
        )

        # Increment trailing indices
        shadow_short_trailing_idx += 1
        shadow_long_trailing_idx += 1
        near_trailing_idx += 1
        far_trailing_idx += 1
        body_long_trailing_idx += 1

    return out_integer


def belt_hold(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = max(setting_body_long.avg_period, setting_shadow_very_short.avg_period)

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx + 1)
        return out_integer

    # Initialize period totals and trailing indices
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - setting_body_long.avg_period
    shadow_very_short_period_total = 0.0
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period

    # Calculate initial period totals
    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total += es.range_of(setting_body_long, i)

    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total += es.range_of(setting_shadow_very_short, i)

    # Proceed with the calculation for the requested range
    for i in range(start_idx, es.len()):
        if es.real_body(i) > es.average(
            setting_body_long, body_long_period_total, i
        ) and (  # long body
            (
                es.candle_color(i).is_white()
                and es.lower_shadow(i)  # white body and very short lower shadow
                < es.average(
                    setting_shadow_very_short, shadow_very_short_period_total, i
                )
            )
            or (
                es.candle_color(i).is_black()
                and es.upper_shadow(i)  # black body and very short upper shadow
                < es.average(
                    setting_shadow_very_short, shadow_very_short_period_total, i
                )
            )
        ):
            out_integer[i] = int(es.candle_color(i)) * 100

        # Update the period totals for the next iteration
        body_long_period_total += es.range_of(setting_body_long, i) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        shadow_very_short_period_total += es.range_of(
            setting_shadow_very_short, i
        ) - es.range_of(setting_shadow_very_short, shadow_very_short_trailing_idx)

        # Increment trailing indices
        body_long_trailing_idx += 1
        shadow_very_short_trailing_idx += 1

    return out_integer


def break_away(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = setting_body_long.avg_period + 4

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Initialize period totals and trailing index
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - setting_body_long.avg_period

    # Calculate initial period total
    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total += es.range_of(setting_body_long, i - 4)

    # Proceed with the calculation for the requested range
    for i in range(start_idx, es.len()):
        if (
            es.real_body(i - 4)
            > es.average(setting_body_long, body_long_period_total, i - 4)
            and es.candle_color(i - 4) == es.candle_color(i - 3)  # 1st long
            and es.candle_color(i - 3)  # 1st, 2nd, 4th same color, 5th opposite
            == es.candle_color(i - 1)
            and es.candle_color(i - 1) == -es.candle_color(i)
            and (
                (
                    es.candle_color(i - 4) == CandleColor.BLACK
                    and es.real_body_gap_down(i - 3, i - 4)  # when 1st is black:
                    and es.high(i - 2) < es.high(i - 3)  # 2nd gaps down
                    and es.low(i - 2) < es.low(i - 3)
                    and es.high(i - 1)  # 3rd has lower high and low than 2nd
                    < es.high(i - 2)
                    and es.low(i - 1) < es.low(i - 2)
                    and es.close(i)  # 4th has lower high and low than 3rd
                    > es.open(i - 3)
                    and es.close(i) < es.close(i - 4)
                )
                or (  # 5th closes inside the gap
                    es.candle_color(i - 4) == CandleColor.WHITE
                    and es.real_body_gap_up(i - 3, i - 4)  # when 1st is white:
                    and es.high(i - 2) > es.high(i - 3)  # 2nd gaps up
                    and es.low(i - 2) > es.low(i - 3)
                    and es.high(i - 1)  # 3rd has higher high and low than 2nd
                    > es.high(i - 2)
                    and es.low(i - 1) > es.low(i - 2)
                    and es.close(i)  # 4th has higher high and low than 3rd
                    < es.open(i - 3)
                    and es.close(i) > es.close(i - 4)
                )
            )
        ):  # 5th closes inside the gap
            out_integer[i] = int(es.candle_color(i)) * 100

        # Update the period total for the next iteration
        body_long_period_total += es.range_of(setting_body_long, i - 4) - es.range_of(
            setting_body_long, body_long_trailing_idx - 4
        )
        body_long_trailing_idx += 1

    return out_integer


def closing_marubozu(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = max(setting_body_long.avg_period, setting_shadow_very_short.avg_period)

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    body_long_period_total = 0.0
    body_long_trailing_idx = start_idx - setting_body_long.avg_period
    shadow_very_short_period_total = 0.0
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period

    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total += es.range_of(setting_body_long, i)

    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total += es.range_of(setting_shadow_very_short, i)

    # Proceed with the calculation for the requested range.
    # Must have:
    # - long white (black) real body
    # - no or very short upper (lower) shadow
    for i in range(start_idx, es.len()):
        if es.real_body(i) > es.average(
            setting_body_long, body_long_period_total, i
        ) and (  # long body
            (
                es.candle_color(i) == CandleColor.WHITE
                and es.upper_shadow(i)
                < es.average(
                    setting_shadow_very_short, shadow_very_short_period_total, i
                )
            )
            or (
                es.candle_color(i) == CandleColor.BLACK
                and es.lower_shadow(i)
                < es.average(
                    setting_shadow_very_short, shadow_very_short_period_total, i
                )
            )
        ):
            out_integer[i] = int(es.candle_color(i)) * 100

        # Add the current range and subtract the first range
        body_long_period_total += es.range_of(setting_body_long, i) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        shadow_very_short_period_total += es.range_of(
            setting_shadow_very_short, i
        ) - es.range_of(setting_shadow_very_short, shadow_very_short_trailing_idx)
        body_long_trailing_idx += 1
        shadow_very_short_trailing_idx += 1

    return out_integer


def conceal_baby_swallow(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = setting_shadow_very_short.avg_period + 3

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    shadow_very_short_period_total = [0.0] * 4
    shadow_very_short_trailing_idx = start_idx - setting_shadow_very_short.avg_period

    for i in range(shadow_very_short_trailing_idx, start_idx):
        shadow_very_short_period_total[3] += es.range_of(
            setting_shadow_very_short, i - 3
        )
        shadow_very_short_period_total[2] += es.range_of(
            setting_shadow_very_short, i - 2
        )
        shadow_very_short_period_total[1] += es.range_of(
            setting_shadow_very_short, i - 1
        )

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 3).is_black()
            and es.candle_color(i - 2).is_black()  # 1st black
            and es.candle_color(i - 1).is_black()  # 2nd black
            and es.candle_color(i).is_black()  # 3rd black
            and  # 4th black
            # 1st: marubozu
            es.lower_shadow(i - 3)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[3], i - 3
            )
            and es.upper_shadow(i - 3)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[3], i - 3
            )
            and
            # 2nd: marubozu
            es.lower_shadow(i - 2)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[2], i - 2
            )
            and es.upper_shadow(i - 2)
            < es.average(
                setting_shadow_very_short, shadow_very_short_period_total[2], i - 2
            )
            and es.real_body_gap_down(i - 1, i - 2)
            and  # 3rd: opens gapping down
            # and HAS an upper shadow
            es.upper_shadow(i - 1)
            > es.average(
                setting_shadow_very_short, shadow_very_short_period_total[1], i - 1
            )
            and es.high(i - 1) > es.close(i - 2)
            and es.high(i) > es.high(i - 1)  # that extends into the prior body
            and es.low(i) < es.low(i - 1)
        ):  # 4th: engulfs the 3rd including the shadows
            out_integer[i] = 100

        # Add the current range and subtract the first range
        for tot_idx in range(3, 0, -1):
            shadow_very_short_period_total[tot_idx] += es.range_of(
                setting_shadow_very_short, i - tot_idx
            ) - es.range_of(
                setting_shadow_very_short, shadow_very_short_trailing_idx - tot_idx
            )
        shadow_very_short_trailing_idx += 1

    return out_integer


def doji(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = setting_body_doji.avg_period

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    body_doji_period_total = 0.0
    body_doji_trailing_idx = start_idx - setting_body_doji.avg_period

    for i in range(body_doji_trailing_idx, start_idx):
        body_doji_period_total += es.range_of(setting_body_doji, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if es.real_body(i) <= es.average(setting_body_doji, body_doji_period_total, i):
            out_integer[i] = 100
        else:
            out_integer[i] = 0

        # Add the current range and subtract the first range
        body_doji_period_total += es.range_of(setting_body_doji, i) - es.range_of(
            setting_body_doji, body_doji_trailing_idx
        )
        body_doji_trailing_idx += 1

    return out_integer


def doji_star(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = max(setting_body_doji.avg_period, setting_body_long.avg_period) + 1

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    body_long_period_total = 0.0
    body_doji_period_total = 0.0
    body_long_trailing_idx = start_idx - 1 - setting_body_long.avg_period
    body_doji_trailing_idx = start_idx - setting_body_doji.avg_period

    for i in range(body_long_trailing_idx, start_idx - 1):
        body_long_period_total += es.range_of(setting_body_long, i)

    for i in range(body_doji_trailing_idx, start_idx):
        body_doji_period_total += es.range_of(setting_body_doji, i)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.real_body(i - 1)
            > es.average(setting_body_long, body_long_period_total, i - 1)
            and es.real_body(i)  # 1st: long real body
            <= es.average(setting_body_doji, body_doji_period_total, i)
            and (  # 2nd: doji
                (es.candle_color(i - 1).is_white() and es.real_body_gap_up(i, i - 1))
                or (  # gaps up if 1st is white
                    es.candle_color(i - 1).is_black()
                    and es.real_body_gap_down(i, i - 1)
                )
            )
        ):  # or down if 1st is black
            out_integer[i] = -int(es.candle_color(i - 1)) * 100
        else:
            out_integer[i] = 0

        # Add the current range and subtract the first range
        body_long_period_total += es.range_of(setting_body_long, i - 1) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        body_doji_period_total += es.range_of(setting_body_doji, i) - es.range_of(
            setting_body_doji, body_doji_trailing_idx
        )
        body_long_trailing_idx += 1
        body_doji_trailing_idx += 1

    return out_integer


def evening_star(series: Series, penetration: float) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    if penetration == DefaultFloat64:
        penetration = 0.3
    elif penetration < 0.0 or penetration > 3.0e37:
        logging.warning("penetration out of range")
        return out_integer

    # Identify the minimum number of price bars needed to calculate at least one output.
    start_idx = max(setting_body_short.avg_period, setting_body_long.avg_period) + 2

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    body_long_period_total = 0.0
    body_short_period_total = 0.0
    body_short_period_total2 = 0.0
    body_long_trailing_idx = start_idx - 2 - setting_body_long.avg_period
    body_short_trailing_idx = start_idx - 1 - setting_body_short.avg_period

    for i in range(body_long_trailing_idx, start_idx - 2):
        body_long_period_total += es.range_of(setting_body_long, i)
    for i in range(body_short_trailing_idx, start_idx - 1):
        body_short_period_total += es.range_of(setting_body_short, i)
        body_short_period_total2 += es.range_of(setting_body_short, i + 1)

    # Proceed with the calculation for the requested range.
    for i in range(start_idx, es.len()):
        if (
            es.real_body(i - 2)
            > es.average(setting_body_long, body_long_period_total, i - 2)
            and es.candle_color(i - 2).is_white()  # 1st: long
            and es.real_body(i - 1)  # white
            <= es.average(setting_body_short, body_short_period_total, i - 1)
            and es.real_body_gap_up(i - 1, i - 2)  # 2nd: short
            and es.real_body(i)  # gapping up
            > es.average(setting_body_short, body_short_period_total2, i)
            and es.candle_color(i).is_black()  # 3rd: longer than short
            and es.close(i)  # black real body
            < es.close(i - 2) - es.real_body(i - 2) * penetration
        ):  # closing well within 1st rb
            out_integer[i] = -100

        # Add the current range and subtract the first range
        body_long_period_total += es.range_of(setting_body_long, i - 2) - es.range_of(
            setting_body_long, body_long_trailing_idx
        )
        body_short_period_total += es.range_of(setting_body_short, i - 1) - es.range_of(
            setting_body_short, body_short_trailing_idx
        )
        body_short_period_total2 += es.range_of(setting_body_short, i) - es.range_of(
            setting_body_short, body_short_trailing_idx + 1
        )
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1

    return out_integer


def matching_low(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = setting_equal.avg_period + 1

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    equal_period_total = 0.0
    equal_trailing_idx = start_idx - setting_equal.avg_period

    for i in range(equal_trailing_idx, start_idx):
        equal_period_total += es.range_of(setting_equal, i - 1)

    # Proceed with the calculation for the requested range.
    # Must have:
    # - first candle: black candle
    # - second candle: black candle with the close equal to the previous close
    # The meaning of "equal" is specified with TA_SetCandleSettings
    # out_integer is always positive (1 to 100): matching low is always bullish;
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 1).is_black()
            and es.candle_color(i).is_black()  # first black
            and es.close(i)  # second black
            <= es.close(i - 1) + es.average(setting_equal, equal_period_total, i - 1)
            and es.close(i)  # 1st and 2nd same close
            >= es.close(i - 1) - es.average(setting_equal, equal_period_total, i - 1)
        ):  # within range
            out_integer[i] = 100

        # Add the current range and subtract the first range: this is done after the pattern recognition
        # when avgPeriod is not 0, that means "compare with the previous candles" (it excludes the current candle)
        equal_period_total += es.range_of(setting_equal, i - 1) - es.range_of(
            setting_equal, equal_trailing_idx - 1
        )
        equal_trailing_idx += 1

    return out_integer


def piercing(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = setting_body_long.avg_period + 1

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    body_long_period_total = [0.0, 0.0]
    body_long_trailing_idx = start_idx - setting_body_long.avg_period

    for i in range(body_long_trailing_idx, start_idx):
        body_long_period_total[1] += es.range_of(setting_body_long, i - 1)
        body_long_period_total[0] += es.range_of(setting_body_long, i)

    # Proceed with the calculation for the requested range.
    # Must have:
    # - first candle: long black candle
    # - second candle: long white candle with open below previous day low and close at least at 50% of previous day real body
    # The meaning of "long" is specified with TA_SetCandleSettings
    # out_integer is positive (1 to 100): piercing pattern is always bullish
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 1).is_black()
            and es.real_body(i - 1)  # 1st: black
            > es.average(setting_body_long, body_long_period_total[1], i - 1)
            and es.candle_color(i).is_white()  # long
            and es.real_body(i)  # 2nd: white
            > es.average(setting_body_long, body_long_period_total[0], i)
            and es.open(i) < es.low(i - 1)  # long
            and es.close(i) < es.open(i - 1)  # open below prior low
            and es.close(i)  # close within prior body
            > es.close(i - 1) + es.real_body(i - 1) * 0.5
        ):  # above midpoint
            out_integer[i] = 100

        # Add the current range and subtract the first range: this is done after the pattern recognition
        # when avgPeriod is not 0, that means "compare with the previous candles" (it excludes the current candle)
        for tot_idx in range(2):
            body_long_period_total[tot_idx] += es.range_of(
                setting_body_long, i - tot_idx
            ) - es.range_of(setting_body_long, body_long_trailing_idx - tot_idx)
        body_long_trailing_idx += 1

    return out_integer


def stick_sandwich(series: Series) -> List[int]:
    es = EnhancedSeries(series)
    out_integer = [0] * es.len()

    # Identify the minimum number of price bars needed to calculate at least one output.
    # Move up the start index if there is not enough initial data.
    start_idx = setting_equal.avg_period + 2

    # Make sure there is still something to evaluate.
    if start_idx >= es.len():
        logging.warning("Too few input len(%d) want(%d)", es.len(), start_idx)
        return out_integer

    # Do the calculation using tight loops.
    # Add up the initial period, except for the last value.
    equal_period_total = 0.0
    equal_trailing_idx = start_idx - setting_equal.avg_period

    for i in range(equal_trailing_idx, start_idx):
        equal_period_total += es.range_of(setting_equal, i - 2)

    # Proceed with the calculation for the requested range.
    # Must have:
    # - first candle: black candle
    # - second candle: white candle that trades only above the prior close (low > prior close)
    # - third candle: black candle with the close equal to the first candle's close
    # The meaning of "equal" is specified with TA_SetCandleSettings
    # out_integer is always positive (1 to 100): stick sandwich is always bullish;
    for i in range(start_idx, es.len()):
        if (
            es.candle_color(i - 2).is_black()
            and es.candle_color(i - 1).is_white()  # first black
            and es.candle_color(i).is_black()  # second white
            and es.low(i - 1) > es.close(i - 2)  # third black
            and es.close(i)  # 2nd low > prior close
            <= es.close(i - 2) + es.average(setting_equal, equal_period_total, i - 2)
            and es.close(i)  # 1st and 3rd same close
            >= es.close(i - 2) - es.average(setting_equal, equal_period_total, i - 2)
        ):  # within range
            out_integer[i] = 100

        # Add the current range and subtract the first range: this is done after the pattern recognition
        # when avgPeriod is not 0, that means "compare with the previous candles" (it excludes the current candle)
        equal_period_total += es.range_of(setting_equal, i - 2) - es.range_of(
            setting_equal, equal_trailing_idx - 2
        )
        equal_trailing_idx += 1

    return out_integer
