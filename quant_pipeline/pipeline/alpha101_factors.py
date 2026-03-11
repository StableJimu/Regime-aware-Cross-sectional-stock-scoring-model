# -*- coding: utf-8 -*-
"""
Alpha 101 (WorldQuant) factor implementations based on Appendix A.
Only alphas requiring OHLCV and VWAP/ADV are included.
Alphas requiring industry neutralization or market cap are omitted.

VWAP handling:
If `vwap` column is not present, we use a proxy: (high + low + close) / 3.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .factor_processing import FactorRegistry


def _floor_window(d: float) -> int:
    return int(np.floor(d))

def _rolling_apply(s: pd.Series, n: int, fn) -> pd.Series:
    def _apply(x: pd.Series) -> pd.Series:
        return fn(x.rolling(n, min_periods=n))
    return s.groupby(level="ticker", group_keys=False).apply(_apply)


def _group_ticker(s: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    return s.groupby(level="ticker")


def _rank_xs(s: pd.Series) -> pd.Series:
    return s.groupby(level="date").rank(pct=True)


def _scale_xs(s: pd.Series, a: float = 1.0) -> pd.Series:
    def _scale_one(x: pd.Series) -> pd.Series:
        denom = x.abs().sum()
        if denom == 0 or np.isnan(denom):
            return x * 0.0
        return x * (a / denom)
    return s.groupby(level="date", group_keys=False).apply(_scale_one)


def _delay(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    return _group_ticker(s).shift(n)


def _delta(s: pd.Series, d: float) -> pd.Series:
    return s - _delay(s, d)


def _ts_min(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    return _rolling_apply(s, n, lambda r: r.min())


def _ts_max(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    return _rolling_apply(s, n, lambda r: r.max())


def _ts_argmax(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _argmax(x: pd.Series) -> float:
        return float(np.argmax(x.values))
    return _rolling_apply(s, n, lambda r: r.apply(_argmax, raw=False))


def _ts_argmin(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _argmin(x: pd.Series) -> float:
        return float(np.argmin(x.values))
    return _rolling_apply(s, n, lambda r: r.apply(_argmin, raw=False))


def _ts_rank(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _rank_last(x: pd.Series) -> float:
        return float(x.rank(pct=True).iloc[-1])
    return _rolling_apply(s, n, lambda r: r.apply(_rank_last, raw=False))


def _sum(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    return _rolling_apply(s, n, lambda r: r.sum())


def _product(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _prod(x: pd.Series) -> float:
        return float(np.prod(x.values))
    return _rolling_apply(s, n, lambda r: r.apply(_prod, raw=False))


def _stddev(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    return _rolling_apply(s, n, lambda r: r.std())


def _correlation(x: pd.Series, y: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _corr(g: pd.Series) -> pd.Series:
        y_g = y.loc[g.index]
        return g.rolling(n, min_periods=n).corr(y_g)
    return x.groupby(level="ticker", group_keys=False).apply(_corr)


def _covariance(x: pd.Series, y: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    def _cov(g: pd.Series) -> pd.Series:
        y_g = y.loc[g.index]
        return g.rolling(n, min_periods=n).cov(y_g)
    return x.groupby(level="ticker", group_keys=False).apply(_cov)


def _decay_linear(s: pd.Series, d: float) -> pd.Series:
    n = _floor_window(d)
    weights = np.arange(1, n + 1, dtype=float)
    weights = weights / weights.sum()
    def _wma(x: pd.Series) -> float:
        return float(np.dot(x.values, weights))
    return _rolling_apply(s, n, lambda r: r.apply(_wma, raw=False))


def _signedpower(x: pd.Series, a: float) -> pd.Series:
    return np.sign(x) * (np.abs(x) ** a)


def _get_series(panel: pd.DataFrame, col: str) -> pd.Series:
    return panel[col].astype(float)


def _returns(panel: pd.DataFrame) -> pd.Series:
    close = _get_series(panel, "close")
    return close.groupby(level="ticker", group_keys=False).apply(lambda s: np.log(s).diff())


def _vwap(panel: pd.DataFrame) -> pd.Series:
    if "vwap" in panel.columns:
        return _get_series(panel, "vwap")
    high = _get_series(panel, "high")
    low = _get_series(panel, "low")
    close = _get_series(panel, "close")
    return (high + low + close) / 3.0


def _adv(panel: pd.DataFrame, d: float) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    dollar_vol = close * volume
    return _sum(dollar_vol, d) / _floor_window(d)


def _bool_to_float(x: pd.Series) -> pd.Series:
    return x.astype(float)


# -------------------------
# Alpha implementations
# -------------------------

def alpha_001(panel: pd.DataFrame, _: Dict) -> pd.Series:
    returns = _returns(panel)
    close = _get_series(panel, "close")
    x = np.where(returns < 0, _stddev(returns, 20), close)
    x = pd.Series(x, index=returns.index)
    s = _ts_argmax(_signedpower(x, 2.0), 5)
    return _rank_xs(s) - 0.5


def alpha_002(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    open_ = _get_series(panel, "open")
    volume = _get_series(panel, "volume")
    x = _rank_xs(_delta(np.log(volume), 2))
    y = _rank_xs((close - open_) / open_)
    return -1.0 * _correlation(x, y, 6)


def alpha_003(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    volume = _get_series(panel, "volume")
    return -1.0 * _correlation(_rank_xs(open_), _rank_xs(volume), 10)


def alpha_004(panel: pd.DataFrame, _: Dict) -> pd.Series:
    low = _get_series(panel, "low")
    return -1.0 * _ts_rank(_rank_xs(low), 9)


def alpha_005(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    close = _get_series(panel, "close")
    vwap = _vwap(panel)
    return _rank_xs(open_ - (_sum(vwap, 10) / 10)) * (-1.0 * np.abs(_rank_xs(close - vwap)))


def alpha_006(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    volume = _get_series(panel, "volume")
    return -1.0 * _correlation(open_, volume, 10)


def alpha_007(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    vol_dollar = close * volume
    cond = adv20 < vol_dollar
    s = (-1.0 * _ts_rank(np.abs(_delta(close, 7)), 60)) * np.sign(_delta(close, 7))
    return pd.Series(np.where(cond, s, -1.0), index=close.index)


def alpha_008(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    returns = _returns(panel)
    x = _sum(open_, 5) * _sum(returns, 5)
    return -1.0 * _rank_xs(x - _delay(x, 10))


def alpha_009(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    d = _delta(close, 1)
    return pd.Series(
        np.where(
            0 < _ts_min(d, 5),
            d,
            np.where(_ts_max(d, 5) < 0, d, -1.0 * d),
        ),
        index=close.index,
    )


def alpha_010(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    d = _delta(close, 1)
    s = np.where(0 < _ts_min(d, 4), d, np.where(_ts_max(d, 4) < 0, d, -1.0 * d))
    return _rank_xs(pd.Series(s, index=close.index))


def alpha_011(panel: pd.DataFrame, _: Dict) -> pd.Series:
    vwap = _vwap(panel)
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    x = _rank_xs(_ts_max(vwap - close, 3)) + _rank_xs(_ts_min(vwap - close, 3))
    return x * _rank_xs(_delta(volume, 3))


def alpha_012(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    return np.sign(_delta(volume, 1)) * (-1.0 * _delta(close, 1))


def alpha_013(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    return -1.0 * _rank_xs(_covariance(_rank_xs(close), _rank_xs(volume), 5))


def alpha_014(panel: pd.DataFrame, _: Dict) -> pd.Series:
    returns = _returns(panel)
    open_ = _get_series(panel, "open")
    volume = _get_series(panel, "volume")
    return (-1.0 * _rank_xs(_delta(returns, 3))) * _correlation(open_, volume, 10)


def alpha_015(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    return -1.0 * _sum(_rank_xs(_correlation(_rank_xs(high), _rank_xs(volume), 3)), 3)


def alpha_016(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    return -1.0 * _rank_xs(_covariance(_rank_xs(high), _rank_xs(volume), 5))


def alpha_017(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    return (
        (-1.0 * _rank_xs(_ts_rank(close, 10)))
        * _rank_xs(_delta(_delta(close, 1), 1))
        * _rank_xs(_ts_rank(volume / adv20, 5))
    )


def alpha_018(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    open_ = _get_series(panel, "open")
    x = _stddev(np.abs(close - open_), 5) + (close - open_)
    return -1.0 * _rank_xs(x + _correlation(close, open_, 10))


def alpha_019(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    returns = _returns(panel)
    x = (close - _delay(close, 7)) + _delta(close, 7)
    return (-1.0 * np.sign(x)) * (1.0 + _rank_xs(1.0 + _sum(returns, 250)))


def alpha_020(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    high = _get_series(panel, "high")
    close = _get_series(panel, "close")
    low = _get_series(panel, "low")
    return (
        (-1.0 * _rank_xs(open_ - _delay(high, 1)))
        * _rank_xs(open_ - _delay(close, 1))
        * _rank_xs(open_ - _delay(low, 1))
    )


def alpha_021(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    vol_dollar = close * volume
    s1 = (_sum(close, 8) / 8) + _stddev(close, 8)
    s2 = _sum(close, 2) / 2
    cond1 = s1 < s2
    cond2 = s2 < ((_sum(close, 8) / 8) - _stddev(close, 8))
    cond3 = (vol_dollar / adv20) >= 1
    out = np.where(cond1, -1.0, np.where(cond2, 1.0, np.where(cond3, 1.0, -1.0)))
    return pd.Series(out, index=close.index)


def alpha_022(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    close = _get_series(panel, "close")
    return -1.0 * (_delta(_correlation(high, volume, 5), 5) * _rank_xs(_stddev(close, 20)))


def alpha_023(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    cond = (_sum(high, 20) / 20) < high
    return pd.Series(np.where(cond, -1.0 * _delta(high, 2), 0.0), index=high.index)


def alpha_024(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    cond = (_delta((_sum(close, 100) / 100), 100) / _delay(close, 100)) <= 0.05
    out = np.where(cond, -1.0 * (close - _ts_min(close, 100)), -1.0 * _delta(close, 3))
    return pd.Series(out, index=close.index)


def alpha_025(panel: pd.DataFrame, _: Dict) -> pd.Series:
    returns = _returns(panel)
    adv20 = _adv(panel, 20)
    vwap = _vwap(panel)
    high = _get_series(panel, "high")
    close = _get_series(panel, "close")
    return _rank_xs(((-1.0 * returns) * adv20) * vwap * (high - close))


def alpha_026(panel: pd.DataFrame, _: Dict) -> pd.Series:
    volume = _get_series(panel, "volume")
    high = _get_series(panel, "high")
    return -1.0 * _ts_max(_correlation(_ts_rank(volume, 5), _ts_rank(high, 5), 5), 3)


def alpha_027(panel: pd.DataFrame, _: Dict) -> pd.Series:
    volume = _get_series(panel, "volume")
    vwap = _vwap(panel)
    x = _sum(_correlation(_rank_xs(volume), _rank_xs(vwap), 6), 2) / 2.0
    return pd.Series(np.where(_rank_xs(x) > 0.5, -1.0, 1.0), index=volume.index)


def alpha_028(panel: pd.DataFrame, _: Dict) -> pd.Series:
    adv20 = _adv(panel, 20)
    low = _get_series(panel, "low")
    high = _get_series(panel, "high")
    close = _get_series(panel, "close")
    return _scale_xs(_correlation(adv20, low, 5) + ((high + low) / 2) - close)


def alpha_029(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    returns = _returns(panel)
    x = _rank_xs(_delta((close - 1), 5))
    x = _rank_xs(-1.0 * x)
    x = _rank_xs(_ts_min(_rank_xs(_rank_xs(x)), 2))
    x = _sum(x, 1)
    x = np.log(x)
    x = _scale_xs(x)
    x = _rank_xs(_rank_xs(x))
    x = _product(x, 5)
    return _ts_rank(_delay(-1.0 * returns, 6), 5) + _ts_min(x, 5)


def alpha_030(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    x = np.sign(close - _delay(close, 1)) + np.sign(_delay(close, 1) - _delay(close, 2)) + np.sign(_delay(close, 2) - _delay(close, 3))
    return ((1.0 - _rank_xs(x)) * _sum(volume, 5)) / _sum(volume, 20)


def alpha_031(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    adv20 = _adv(panel, 20)
    low = _get_series(panel, "low")
    part1 = _rank_xs(_rank_xs(_rank_xs(_decay_linear(-1.0 * _rank_xs(_rank_xs(_delta(close, 10))), 10))))
    part2 = _rank_xs(-1.0 * _delta(close, 3))
    part3 = np.sign(_scale_xs(_correlation(adv20, low, 12)))
    return part1 + part2 + part3


def alpha_032(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    vwap = _vwap(panel)
    return _scale_xs((_sum(close, 7) / 7) - close) + (20.0 * _scale_xs(_correlation(vwap, _delay(close, 5), 230)))


def alpha_033(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    close = _get_series(panel, "close")
    return _rank_xs(-1.0 * (1.0 - (open_ / close)) ** 1)


def alpha_034(panel: pd.DataFrame, _: Dict) -> pd.Series:
    returns = _returns(panel)
    close = _get_series(panel, "close")
    return _rank_xs((1.0 - _rank_xs(_stddev(returns, 2) / _stddev(returns, 5))) + (1.0 - _rank_xs(_delta(close, 1))))


def alpha_035(panel: pd.DataFrame, _: Dict) -> pd.Series:
    volume = _get_series(panel, "volume")
    close = _get_series(panel, "close")
    high = _get_series(panel, "high")
    returns = _returns(panel)
    return _ts_rank(volume, 32) * (1.0 - _ts_rank((close + high) - _get_series(panel, "low"), 16)) * (1.0 - _ts_rank(returns, 32))


def alpha_036(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    open_ = _get_series(panel, "open")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    vwap = _vwap(panel)
    term1 = 2.21 * _rank_xs(_correlation((close - open_), _delay(volume, 1), 15))
    term2 = 0.7 * _rank_xs(open_ - close)
    term3 = 0.73 * _rank_xs(_ts_rank(_delay(-1.0 * _returns(panel), 6), 5))
    term4 = _rank_xs(np.abs(_correlation(vwap, adv20, 6)))
    term5 = 0.6 * _rank_xs((_sum(close, 200) / 200 - open_) * (close - open_))
    return term1 + term2 + term3 + term4 + term5


def alpha_037(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    close = _get_series(panel, "close")
    return _rank_xs(_correlation(_delay(open_ - close, 1), close, 200)) + _rank_xs(open_ - close)


def alpha_038(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    open_ = _get_series(panel, "open")
    return (-1.0 * _rank_xs(_ts_rank(close, 10))) * _rank_xs(close / open_)


def alpha_039(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    returns = _returns(panel)
    term = _delta(close, 7) * (1.0 - _rank_xs(_decay_linear(volume / adv20, 9)))
    return (-1.0 * _rank_xs(term)) * (1.0 + _rank_xs(_sum(returns, 250)))


def alpha_040(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    return (-1.0 * _rank_xs(_stddev(high, 10))) * _correlation(high, volume, 10)


def alpha_041(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    low = _get_series(panel, "low")
    vwap = _vwap(panel)
    return np.sqrt(high * low) - vwap


def alpha_042(panel: pd.DataFrame, _: Dict) -> pd.Series:
    vwap = _vwap(panel)
    close = _get_series(panel, "close")
    return _rank_xs(vwap - close) / _rank_xs(vwap + close)


def alpha_043(panel: pd.DataFrame, _: Dict) -> pd.Series:
    volume = _get_series(panel, "volume")
    close = _get_series(panel, "close")
    adv20 = _adv(panel, 20)
    return _ts_rank(volume / adv20, 20) * _ts_rank(-1.0 * _delta(close, 7), 8)


def alpha_044(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    return -1.0 * _correlation(high, _rank_xs(volume), 5)


def alpha_045(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    return -1.0 * (
        _rank_xs(_sum(_delay(close, 5), 20) / 20)
        * _correlation(close, volume, 2)
        * _rank_xs(_correlation(_sum(close, 5), _sum(close, 20), 2))
    )


def alpha_046(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    x = ((_delay(close, 20) - _delay(close, 10)) / 10) - ((_delay(close, 10) - close) / 10)
    return pd.Series(
        np.where(x > 0.25, -1.0, np.where(x < 0, 1.0, -1.0 * (close - _delay(close, 1)))),
        index=close.index,
    )


def alpha_047(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    volume = _get_series(panel, "volume")
    adv20 = _adv(panel, 20)
    high = _get_series(panel, "high")
    vwap = _vwap(panel)
    return (( _rank_xs(1.0 / close) * volume) / adv20) * ((high * _rank_xs(high - close)) / (_sum(high, 5) / 5)) - _rank_xs(vwap - _delay(vwap, 5))


def alpha_049(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    x = ((_delay(close, 20) - _delay(close, 10)) / 10) - ((_delay(close, 10) - close) / 10)
    return pd.Series(np.where(x < -0.1, 1.0, -1.0 * (close - _delay(close, 1))), index=close.index)


def alpha_050(panel: pd.DataFrame, _: Dict) -> pd.Series:
    volume = _get_series(panel, "volume")
    vwap = _vwap(panel)
    return -1.0 * _ts_max(_rank_xs(_correlation(_rank_xs(volume), _rank_xs(vwap), 5)), 5)


def alpha_051(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    x = ((_delay(close, 20) - _delay(close, 10)) / 10) - ((_delay(close, 10) - close) / 10)
    return pd.Series(np.where(x < -0.05, 1.0, -1.0 * (close - _delay(close, 1))), index=close.index)


def alpha_052(panel: pd.DataFrame, _: Dict) -> pd.Series:
    low = _get_series(panel, "low")
    volume = _get_series(panel, "volume")
    returns = _returns(panel)
    part1 = (-1.0 * _ts_min(low, 5)) + _delay(_ts_min(low, 5), 5)
    part2 = _rank_xs((_sum(returns, 240) - _sum(returns, 20)) / 220)
    part3 = _ts_rank(volume, 5)
    return part1 * part2 * part3


def alpha_053(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    low = _get_series(panel, "low")
    high = _get_series(panel, "high")
    return -1.0 * _delta(((close - low) - (high - close)) / (close - low), 9)


def alpha_054(panel: pd.DataFrame, _: Dict) -> pd.Series:
    low = _get_series(panel, "low")
    close = _get_series(panel, "close")
    high = _get_series(panel, "high")
    open_ = _get_series(panel, "open")
    return (-1.0 * ((low - close) * (open_ ** 5))) / ((low - high) * (close ** 5))


def alpha_055(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    low = _get_series(panel, "low")
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    x = (close - _ts_min(low, 12)) / (_ts_max(high, 12) - _ts_min(low, 12))
    return -1.0 * _correlation(_rank_xs(x), _rank_xs(volume), 6)


def alpha_057(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    vwap = _vwap(panel)
    return -1.0 * ((close - vwap) / _decay_linear(_rank_xs(_ts_argmax(close, 30)), 2))


def alpha_060(panel: pd.DataFrame, _: Dict) -> pd.Series:
    close = _get_series(panel, "close")
    low = _get_series(panel, "low")
    high = _get_series(panel, "high")
    volume = _get_series(panel, "volume")
    term1 = 2.0 * _scale_xs(_rank_xs((((close - low) - (high - close)) / (high - low)) * volume))
    term2 = _scale_xs(_rank_xs(_ts_argmax(close, 10)))
    return -1.0 * (term1 - term2)


def alpha_061(panel: pd.DataFrame, _: Dict) -> pd.Series:
    vwap = _vwap(panel)
    adv180 = _adv(panel, 180)
    left = _rank_xs(vwap - _ts_min(vwap, 16.1219))
    right = _rank_xs(_correlation(vwap, adv180, 17.9282))
    return _bool_to_float(left < right)


def alpha_062(panel: pd.DataFrame, _: Dict) -> pd.Series:
    vwap = _vwap(panel)
    adv20 = _adv(panel, 20)
    open_ = _get_series(panel, "open")
    high = _get_series(panel, "high")
    low = _get_series(panel, "low")
    left = _rank_xs(_correlation(vwap, _sum(adv20, 22.4101), 9.91009))
    right = _rank_xs((_rank_xs(open_) + _rank_xs(open_)) < (_rank_xs((high + low) / 2) + _rank_xs(high)))
    return -1.0 * _bool_to_float(left < right)


def alpha_064(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    low = _get_series(panel, "low")
    vwap = _vwap(panel)
    adv120 = _adv(panel, 120)
    left = _rank_xs(_correlation(_sum((open_ * 0.178404) + (low * (1 - 0.178404)), 12.7054),
                                 _sum(adv120, 12.7054), 16.6208))
    right = _rank_xs(_delta((((_get_series(panel, "high") + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741))
    return -1.0 * _bool_to_float(left < right)


def alpha_065(panel: pd.DataFrame, _: Dict) -> pd.Series:
    open_ = _get_series(panel, "open")
    vwap = _vwap(panel)
    adv60 = _adv(panel, 60)
    left = _rank_xs(_correlation((open_ * 0.00817205) + (vwap * (1 - 0.00817205)), _sum(adv60, 8.6911), 6.40374))
    right = _rank_xs(open_ - _ts_min(open_, 13.635))
    return -1.0 * _bool_to_float(left < right)


def alpha_066(panel: pd.DataFrame, _: Dict) -> pd.Series:
    vwap = _vwap(panel)
    low = _get_series(panel, "low")
    open_ = _get_series(panel, "open")
    high = _get_series(panel, "high")
    part1 = _rank_xs(_decay_linear(_delta(vwap, 3.51013), 7.23052))
    part2 = _ts_rank(_decay_linear(((low * 0.96633) + (low * (1 - 0.96633)) - vwap) / (open_ - ((high + low) / 2)), 11.4157), 6.72611)
    return -1.0 * (part1 + part2)


def alpha_068(panel: pd.DataFrame, _: Dict) -> pd.Series:
    high = _get_series(panel, "high")
    close = _get_series(panel, "close")
    low = _get_series(panel, "low")
    adv15 = _adv(panel, 15)
    left = _ts_rank(_correlation(_rank_xs(high), _rank_xs(adv15), 8.91644), 13.9333)
    right = _rank_xs(_delta((close * 0.518371) + (low * (1 - 0.518371)), 1.06157))
    return -1.0 * _bool_to_float(left < right)


def register_alpha101(registry: FactorRegistry) -> None:
    """Register Alpha#1-#47, #49-#55, #57, #60-#66, #68."""
    mapping = {
        "alpha_001": alpha_001,
        "alpha_002": alpha_002,
        "alpha_003": alpha_003,
        "alpha_004": alpha_004,
        "alpha_005": alpha_005,
        "alpha_006": alpha_006,
        "alpha_007": alpha_007,
        "alpha_008": alpha_008,
        "alpha_009": alpha_009,
        "alpha_010": alpha_010,
        "alpha_011": alpha_011,
        "alpha_012": alpha_012,
        "alpha_013": alpha_013,
        "alpha_014": alpha_014,
        "alpha_015": alpha_015,
        "alpha_016": alpha_016,
        "alpha_017": alpha_017,
        "alpha_018": alpha_018,
        "alpha_019": alpha_019,
        "alpha_020": alpha_020,
        "alpha_021": alpha_021,
        "alpha_022": alpha_022,
        "alpha_023": alpha_023,
        "alpha_024": alpha_024,
        "alpha_025": alpha_025,
        "alpha_026": alpha_026,
        "alpha_027": alpha_027,
        "alpha_028": alpha_028,
        "alpha_029": alpha_029,
        "alpha_030": alpha_030,
        "alpha_031": alpha_031,
        "alpha_032": alpha_032,
        "alpha_033": alpha_033,
        "alpha_034": alpha_034,
        "alpha_035": alpha_035,
        "alpha_036": alpha_036,
        "alpha_037": alpha_037,
        "alpha_038": alpha_038,
        "alpha_039": alpha_039,
        "alpha_040": alpha_040,
        "alpha_041": alpha_041,
        "alpha_042": alpha_042,
        "alpha_043": alpha_043,
        "alpha_044": alpha_044,
        "alpha_045": alpha_045,
        "alpha_046": alpha_046,
        "alpha_047": alpha_047,
        "alpha_049": alpha_049,
        "alpha_050": alpha_050,
        "alpha_051": alpha_051,
        "alpha_052": alpha_052,
        "alpha_053": alpha_053,
        "alpha_054": alpha_054,
        "alpha_055": alpha_055,
        "alpha_057": alpha_057,
        "alpha_060": alpha_060,
        "alpha_061": alpha_061,
        "alpha_062": alpha_062,
        "alpha_064": alpha_064,
        "alpha_065": alpha_065,
        "alpha_066": alpha_066,
        "alpha_068": alpha_068,
    }
    for name, fn in mapping.items():
        registry.register(name, fn, mode="panel")
