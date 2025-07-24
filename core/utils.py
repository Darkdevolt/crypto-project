```python
import numpy as np
import pandas as pd

def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 365.0
    delta_sec = (index[1] - index[0]).total_seconds()
    if delta_sec == 0:
        return 365.0
    periods_per_day = 86400.0 / delta_sec
    return periods_per_day * 365.0


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def max_drawdown(equity_curve: pd.Series) -> float:
    cummax = equity_curve.cummax()
    dd = (equity_curve / cummax - 1.0).min()
    return dd


def performance_metrics(equity_curve: pd.Series, returns: pd.Series, rf: float = 0.0,
                        periods_per_year: float = 365.0):
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = returns.std() * np.sqrt(periods_per_year)
    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sharpe = (returns.mean() * periods_per_year - rf) / vol if vol > 0 else np.nan
    sortino = (returns.mean() * periods_per_year - rf) / downside if downside > 0 else np.nan
    calmar = (cagr / abs(max_drawdown(equity_curve))) if max_drawdown(equity_curve) < 0 else np.nan
    hit_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    rr = (avg_win / abs(avg_loss)) if avg_loss is not None and avg_loss < 0 else np.nan
    avg_exposure = (returns != 0).mean()
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Vol (ann.)": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_drawdown(equity_curve),
        "Hit Rate": hit_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Avg R/R": rr,
        "Exposure": avg_exposure,
        "N trades": np.nan,
    }


def performance_table(stats: dict) -> pd.DataFrame:
    order = [
        "Total Return", "CAGR", "Vol (ann.)", "Sharpe", "Sortino", "Calmar",
        "Max Drawdown", "Hit Rate", "Avg Win", "Avg Loss", "Avg R/R",
        "Exposure", "N trades"
    ]
    df = pd.DataFrame({"Metric": order, "Value": [stats.get(k, np.nan) for k in order]})
    return df.set_index("Metric")