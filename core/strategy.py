```python
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
from .utils import infer_periods_per_year, compute_atr, performance_metrics

@dataclass
class StrategyConfig:
    # Momentum
    mom_lookback: int = 24 * 7
    ema_fast: int = 50
    ema_slow: int = 200
    signal_smooth: int = 3

    # Vol targeting / levier
    target_vol_annual: float = 0.35
    max_leverage: float = 3.0
    vol_lookback: int = 24 * 7

    # Stops
    atr_period: int = 14
    atr_mult_sl: float = 2.5
    atr_mult_trailing: float = 3.0

    # Money/Risk management
    kill_switch_dd: float = 0.20
    kill_switch_risk_scale: float = 0.3

    # Trading costs (aller + retour)
    fee_bp: float = 5

    # Divers
    allow_shorts: bool = True


class AggressiveVolTargetStrategy:
    def __init__(self, df: pd.DataFrame, config: StrategyConfig):
        self.df = df.copy()
        self.cfg = config
        self._prepare()

    def _prepare(self):
        df = self.df.sort_index()
        cfg = self.cfg
        df["ret"] = df["close"].pct_change().fillna(0.0)

        # Momentum
        df["mom"] = df["close"].pct_change(cfg.mom_lookback)
        signal_mom = np.sign(df["mom"])  # +1 / -1

        # Filtre tendance
        df["ema_fast"] = df["close"].ewm(span=cfg.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=cfg.ema_slow, adjust=False).mean()
        trend = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        # Signal brut
        raw_signal = signal_mom * trend
        if not cfg.allow_shorts:
            raw_signal = np.where(raw_signal > 0, 1.0, 0.0)
        df["signal_raw"] = pd.Series(raw_signal, index=df.index).fillna(0)
        df["signal"] = df["signal_raw"].rolling(cfg.signal_smooth).mean().fillna(0)

        # ATR
        df["atr"] = compute_atr(df, cfg.atr_period)

        # Vol
        periods_per_year = infer_periods_per_year(df.index)
        df["realized_vol"] = df["ret"].rolling(cfg.vol_lookback).std() * np.sqrt(periods_per_year)
        self.periods_per_year = periods_per_year
        self.df = df

    def backtest(self):
        df = self.df
        cfg = self.cfg
        fee = cfg.fee_bp / 1e4

        equity = [1.0]
        pos = 0.0
        entry_price = 0.0
        stop_price = np.nan
        trailing_price = np.nan
        risk_scale = 1.0
        hwm = 1.0
        trades = 0
        position_series = []
        leverage_series = []
        stop_series = []
        trailing_series = []
        dd_series = []
        trade_direction = 0

        for i in range(1, len(df)):
            row_prev = df.iloc[i-1]
            row = df.iloc[i]

            # Kill-switch drawdown
            cur_eq = equity[-1]
            if cur_eq > hwm:
                hwm = cur_eq
            dd = cur_eq / hwm - 1.0
            dd_series.append(dd)
            if dd <= -cfg.kill_switch_dd:
                risk_scale = cfg.kill_switch_risk_scale
            elif cur_eq == hwm:
                risk_scale = 1.0

            # Vol targeting
            if row["realized_vol"] > 0:
                lev = (cfg.target_vol_annual / row["realized_vol"]) * risk_scale
            else:
                lev = 0.0
            lev = np.clip(lev, 0.0, cfg.max_leverage)
            target_pos = row["signal"] * lev

            # Stops si en position
            if pos != 0:
                if trade_direction > 0:
                    if not np.isnan(stop_price) and row["low"] <= stop_price:
                        ret_trade = (stop_price - row_prev["close"]) / row_prev["close"] * pos
                        cur_eq = cur_eq * (1 + ret_trade) * (1 - fee)
                        equity[-1] = cur_eq
                        pos = 0.0
                        stop_price = np.nan
                        trailing_price = np.nan
                        trades += 1
                else:
                    if not np.isnan(stop_price) and row["high"] >= stop_price:
                        ret_trade = (row_prev["close"] - stop_price) / row_prev["close"] * (-pos)
                        cur_eq = cur_eq * (1 + ret_trade) * (1 - fee)
                        equity[-1] = cur_eq
                        pos = 0.0
                        stop_price = np.nan
                        trailing_price = np.nan
                        trades += 1

            # PnL normal
            ret = row["ret"] * pos
            new_eq = equity[-1] * (1 + ret)

            # Rebalancing cost
            if np.sign(target_pos) != np.sign(pos) or (abs(target_pos - pos) > 1e-8):
                size_change = abs(target_pos - pos)
                new_eq *= (1 - fee * size_change)

                # Nouveau trade => stops
                if target_pos != 0 and np.sign(target_pos) != np.sign(pos):
                    entry_price = row["close"]
                    if target_pos > 0:
                        stop_price = entry_price - cfg.atr_mult_sl * row["atr"]
                        trailing_price = entry_price - cfg.atr_mult_trailing * row["atr"]
                        trade_direction = 1
                    else:
                        stop_price = entry_price + cfg.atr_mult_sl * row["atr"]
                        trailing_price = entry_price + cfg.atr_mult_trailing * row["atr"]
                        trade_direction = -1
                    trades += 1

            # Trailing
            if target_pos != 0:
                if trade_direction > 0:
                    new_trailing = row["close"] - cfg.atr_mult_trailing * row["atr"]
                    trailing_price = max(trailing_price, new_trailing)
                    stop_price = max(stop_price, trailing_price)
                else:
                    new_trailing = row["close"] + cfg.atr_mult_trailing * row["atr"]
                    trailing_price = min(trailing_price, new_trailing)
                    stop_price = min(stop_price, trailing_price)

            pos = target_pos
            equity.append(new_eq)
            position_series.append(pos)
            leverage_series.append(abs(pos))
            stop_series.append(stop_price)
            trailing_series.append(trailing_price)

        eq_series = pd.Series(equity, index=df.index)
        pos_series = pd.Series([0.0] + position_series, index=df.index)
        lev_series = pd.Series([0.0] + leverage_series, index=df.index)
        stop_series = pd.Series([np.nan] + stop_series, index=df.index)
        trailing_series = pd.Series([np.nan] + trailing_series, index=df.index)
        dd_series = pd.Series([0.0] + dd_series, index=df.index)

        ret_series = eq_series.pct_change().fillna(0.0)
        stats = performance_metrics(eq_series, ret_series, periods_per_year=self.periods_per_year)
        stats["N trades"] = trades

        out = {
            "equity": eq_series,
            "returns": ret_series,
            "position": pos_series,
            "leverage": lev_series,
            "stop": stop_series,
            "trailing": trailing_series,
            "drawdown": dd_series,
            "stats": stats,
        }
        self.results = out
        return out
