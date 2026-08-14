"""Bridges Kite Connect's missing native weekly interval for this fork.

skytrade-smallcap runs entirely on weekly signals (see 2026-08-14 research:
weekly-resampled signals on the Nifty Smallcap 250 held up comparably to
the parent project's hourly system -- 56-60% win rate, PF 3.5-3.8 -- while
cutting trade frequency ~90x, which matters a lot more for smallcaps'
wider spreads/lower liquidity than it did for the parent's large/midcap
universe). Kite's Historical Data API has no "week" interval
(``infrastructure/kite.py``'s ``_INTERVAL_MAP`` only goes down to "day"),
so this wraps the real day-interval Kite provider and resamples client-side
-- every price is still real Kite day-interval data, nothing synthetic
about the OHLC values themselves, just aggregated into weekly buckets
before ``AlphaEngine`` ever sees them.

Always requests the widest available daily window rather than trusting the
caller's `days` argument (``application/signal_pipeline.py``'s
``_evaluate_symbol`` only asks for 5 days on a routine refresh, which is
Correct for a real "day" interval but would reconstruct only a *fragment*
of the current week if resampled directly -- silently overwriting a
previously-stored, more complete weekly candle on the next upsert). Every
call resamples the full history, so every returned weekly bar -- including
the current, still-forming one -- is always built from its complete set of
daily bars.
"""

import pandas as pd

WEEKLY_INTERVAL = "week"

# Kite's own per-request cap for day-interval history (infrastructure/kite.py's
# _MAX_DAYS_PER_REQUEST["day"]) -- KiteProvider.get_recent_history already
# chunks transparently above this, so requesting it every call is safe, just
# one (possibly multi-chunk) HTTP round trip per symbol -- cheap at this
# fork's once-a-day cadence.
_MIN_DAILY_WINDOW_DAYS = 2000


class WeeklyResamplingProvider:
    """Duck-typed like KiteProvider/YahooProvider (``get_recent_history``),
    so it drops into ``application/signal_pipeline.py``'s existing
    provider-shaped code paths with no changes needed there beyond the wrap
    itself. Only intercepts interval == "week"; anything else passes
    straight through to the wrapped provider unchanged.
    """

    def __init__(
        self,
        base_provider,
        min_daily_window_days: int = _MIN_DAILY_WINDOW_DAYS,
    ) -> None:
        self._base = base_provider
        self._min_daily_window_days = min_daily_window_days

    def get_recent_history(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        if interval != WEEKLY_INTERVAL:
            return self._base.get_recent_history(symbol, interval, days)
        daily = self._base.get_recent_history(
            symbol, "day", max(days, self._min_daily_window_days)
        )
        return resample_to_weekly(daily)


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample a Kite day-interval OHLCV frame (Open/High/Low/Close/Volume,
    tz-aware Datetime index) into weekly bars anchored to Friday close --
    NSE's trading week is Mon-Fri.
    """
    weekly = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna(subset=["Open", "High", "Low", "Close"])
