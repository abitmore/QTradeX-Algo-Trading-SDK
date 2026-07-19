"""
Regression tests for the disk-based candle cache in `qtradex.public.data.Data`.

Covers upstream issue #3 ("[Help Wanted] Debug data caching"): startups with
`end="now"` should not spuriously re-fetch / re-merge candles that are
already fully cached and unchanged, just because `self.end` gets rounded to a
candle boundary that sits ahead of the last candle that could possibly have
closed. Genuinely stale caches must still be topped up correctly.

No network calls are made: `Data.gather_data` is patched out entirely (and
asserted to be called, or not called, depending on the scenario), and the
JSON cache pipe (`qtradex.public.data.json_ipc`) is patched with an in-memory
stand-in so the tests never touch the real `qtradex/common/pipe` cache files
on disk.
"""

import json
import math
import time
import unittest
from unittest import mock

import numpy as np

from qtradex.public.data import Data


def _make_fake_json_ipc(store):
    """
    A tiny in-memory stand-in for `qtradex.common.json_ipc.json_ipc`, so
    tests never touch the real on-disk cache pipe.
    """

    def fake_json_ipc(doc="", text=None, initialize=False, append=False):
        if text is not None:
            store[doc] = text
            return None
        if doc in store:
            return json.loads(store[doc])
        raise FileNotFoundError(doc)

    return fake_json_ipc


def _make_candles(start, count, candle_size, price=100.0):
    unix = [start + i * candle_size for i in range(count)]
    return {
        "unix": unix,
        "open": [price] * count,
        "high": [price + 1] * count,
        "low": [price - 1] * count,
        "close": [price] * count,
        "volume": [10.0] * count,
    }


class DataCacheTest(unittest.TestCase):
    def setUp(self):
        self.candle_size = 3600  # 1 hour, for a fast-running test
        self.exchange = "faketestexchange"
        self.asset = "AAA"
        self.currency = "BBB"
        self.index_key = str(
            (self.exchange, None, self.candle_size, self.asset, self.currency)
        )

        # The most recent candle that could genuinely have finished closing
        # right now.
        now_boundary = math.floor(time.time() / self.candle_size) * self.candle_size
        self.last_closed = now_boundary - self.candle_size

        self.store = {}
        self.json_ipc_patcher = mock.patch(
            "qtradex.public.data.json_ipc", new=_make_fake_json_ipc(self.store)
        )
        self.json_ipc_patcher.start()
        self.addCleanup(self.json_ipc_patcher.stop)

    def _seed_cache(self, first_candle_count=24):
        """
        Seed the fake cache pipe with `first_candle_count` hourly candles
        ending exactly at the last candle that could possibly have closed --
        i.e. a cache that is already fully current.
        """
        begin = self.last_closed - (first_candle_count - 1) * self.candle_size
        candles = _make_candles(begin, first_candle_count, self.candle_size)
        self.store[f"{self.index_key} candles.json"] = json.dumps(candles)
        self.store["data_index.json"] = json.dumps(
            {self.index_key: [begin, self.last_closed]}
        )
        self.store["min_time.json"] = json.dumps({})
        return begin

    def test_up_to_date_cache_does_not_refetch(self):
        """
        The core regression test for upstream issue #3: re-initializing Data
        with end="now" shortly after the cache was already brought fully
        current must not trigger any network fetch, and must not add any
        "new" candles beyond what was genuinely already cached.
        """
        begin = self._seed_cache(first_candle_count=24)

        with mock.patch.object(
            Data,
            "gather_data",
            side_effect=AssertionError(
                "gather_data should not be called when the cache is already current"
            ),
        ) as gather_mock:
            data = Data(
                exchange=self.exchange,
                asset=self.asset,
                currency=self.currency,
                begin=begin,
                end=None,
                candle_size=self.candle_size,
            )

        gather_mock.assert_not_called()
        # exactly the 24 candles that were already cached -- nothing dropped,
        # nothing spuriously duplicated/added.
        self.assertEqual(len(data.raw_candles["unix"]), 24)
        self.assertEqual(int(data.raw_candles["unix"][-1]), self.last_closed)

    def test_repeated_up_to_date_calls_do_not_shrink_cache(self):
        """
        Regression test for a second bug the naive fix would otherwise
        surface: on a pure cache hit, the "crop the incomplete trailing
        candle" logic must not run against already-cached (already correctly
        cropped) data, or the cache would lose one real candle every time
        Data is re-initialized with end="now" and nothing new has closed.
        """
        begin = self._seed_cache(first_candle_count=24)

        with mock.patch.object(
            Data, "gather_data", side_effect=AssertionError("should not fetch")
        ):
            Data(
                exchange=self.exchange,
                asset=self.asset,
                currency=self.currency,
                begin=begin,
                end=None,
                candle_size=self.candle_size,
            )
            cache_after_first = json.loads(
                self.store[f"{self.index_key} candles.json"]
            )

            Data(
                exchange=self.exchange,
                asset=self.asset,
                currency=self.currency,
                begin=begin,
                end=None,
                candle_size=self.candle_size,
            )
            cache_after_second = json.loads(
                self.store[f"{self.index_key} candles.json"]
            )

        self.assertEqual(len(cache_after_first["unix"]), 24)
        self.assertEqual(cache_after_first, cache_after_second)

    def test_genuinely_stale_cache_still_gets_topped_up(self):
        """
        Control test: when the cache is genuinely behind (several real
        candles have closed since it was last updated), a fetch must still
        happen and the new candles must be merged in.
        """
        # cache is 5 candles behind the last closed candle
        stale_last = self.last_closed - 5 * self.candle_size
        begin = stale_last - 23 * self.candle_size
        candles = _make_candles(begin, 24, self.candle_size)
        self.store[f"{self.index_key} candles.json"] = json.dumps(candles)
        self.store["data_index.json"] = json.dumps(
            {self.index_key: [begin, stale_last]}
        )
        self.store["min_time.json"] = json.dumps({})

        # overlap candle + 5 genuinely new ones
        new_candles = _make_candles(stale_last, 6, self.candle_size)

        def fake_gather(self_, candle_size, gather_begin, gather_end, asset, currency):
            return {k: np.array(v) for k, v in new_candles.items()}

        with mock.patch.object(
            Data, "gather_data", autospec=True, side_effect=fake_gather
        ) as gather_mock:
            data = Data(
                exchange=self.exchange,
                asset=self.asset,
                currency=self.currency,
                begin=begin,
                end=None,
                candle_size=self.candle_size,
            )

        gather_mock.assert_called()
        # old candles + 5 genuinely new ones, no duplicates
        unix_values = sorted(int(u) for u in data.raw_candles["unix"])
        self.assertEqual(len(unix_values), len(set(unix_values)))
        self.assertEqual(unix_values[-1], self.last_closed)


if __name__ == "__main__":
    unittest.main()
