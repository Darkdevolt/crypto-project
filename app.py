```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.strategy import StrategyConfig, AggressiveVolTargetStrategy
from core.utils import performance_table

st.set_page_config(page_title="SOL Backtester – Vol Target Momentum", layout="wide")
st.title("SOL Backtester – Vol Target + Momentum + ATR Stops")

st.markdown("""
Stratégie **agressive mais disciplinée** :
- Momentum + filtre EMA
- **Volatility targeting** (levier dynamique)
- **ATR stop & trailing stop**
- **Kill‑switch** sur drawdown
""")

# ---------------------------
# Sidebar – Paramètres
# ---------------------------
st.sidebar.header("Paramètres de la stratégie")

# Core params
mom_lookback = st.sidebar.number_input("Momentum lookback (périodes)", 24*7, 24*30*6, 24*7, step=1)
ema_fast = st.sidebar.number_input("EMA fast", 5, 500, 50, step=1)
ema_slow = st.sidebar.number_input("EMA slow", 10, 1000, 200, step=1)
signal_smooth = st.sidebar.number_input("Lissage du signal", 1, 50, 3, step=1)

# Vol targeting
target_vol_annual = st.sidebar.number_input("Target Vol annualisée", 0.05, 2.0, 0.35, step=0.01)
max_leverage = st.sidebar.number_input("Levier max", 0.0, 10.0, 3.0, step=0.1)
vol_lookback = st.sidebar.number_input("Vol lookback (périodes)", 24, 24*365, 24*7, step=1)

# Stops
atr_period = st.sidebar.number_input("ATR period", 5, 200, 14, step=1)
atr_mult_sl = st.sidebar.number_input("ATR Stop-Loss x", 0.5, 10.0, 2.5, step=0.1)
atr_mult_trailing = st.sidebar.number_input("ATR Trailing x", 0.5, 10.0, 3.0, step=0.1)

# Risk management
kill_switch_dd = st.sidebar.number_input("Kill-switch DD (ex: 0.20 = 20%)", 0.01, 1.0, 0.20, step=0.01)
kill_switch_risk_scale = st.sidebar.number_input("Risk scale après kill-switch", 0.0, 1.0, 0.30, step=0.05)
fee_bp = st.sidebar.number_input("Frais (bps par trade)", 0, 100, 5, step=1)
allow_shorts = st.sidebar.checkbox("Autoriser les shorts", True)

cfg = StrategyConfig(
    mom_lookback=int(mom_lookback),
    ema_fast=int(ema_fast),
    ema_slow=int(ema_slow),
    signal_smooth=int(signal_smooth),
    target_vol_annual=float(target_vol_annual),
    max_leverage=float(max_leverage),
    vol_lookback=int(vol_lookback),
    atr_period=int(atr_period),
    atr_mult_sl=float(atr_mult_sl),
    atr_mult_trailing=float(atr_mult_trailing),
    kill_switch_dd=float(kill_switch_dd),
    kill_switch_risk_scale=float(kill_switch_risk_scale),
    fee_bp=int(fee_bp),
    allow_shorts=allow_shorts,
)

# ---------------------------
# Data input
# ---------------------------
st.sidebar.header("Données")
source = st.sidebar.radio("Source des données", ["Uploader CSV", "Exemple random"], index=0)

if source == "Uploader CSV":
    file = st.sidebar.file_uploader("CSV avec colonnes: timestamp, open, high, low, close, volume", type=["csv"]) 
    if file is not None:
        df = pd.read_csv(file, parse_dates=["timestamp"], index_col="timestamp")
        df = df.sort_index()
    else:
        st.stop()
else:
    # Exemple factice si aucun CSV
    rng = pd.date_range("2023-01-01", periods=24*90, freq="H")
    price = 20 + np.cumsum(np.random.randn(len(rng))) * 0.1
    high = price * (1 + np.random.rand(len(rng))*0.01)
    low = price * (1 - np.random.rand(len(rng))*0.01)
    open_ = price * (1 + (np.random.rand(len(rng)) - 0.5)*0.005)
    volume = np.random.randint(1000, 5000, len(rng))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": price, "volume": volume}, index=rng)

st.success(f"Données chargées : {df.index.min().date()} → {df.index.max().date()} | {len(df)} lignes")

run = st.sidebar.button("▶️ Lancer le backtest")

if run:
    strat = AggressiveVolTargetStrategy(df, cfg)
    res = strat.backtest()

    # Stats table
    st.subheader("📊 Statistiques")
    stats_df = performance_table(res["stats"])  # joli formatage
    st.dataframe(stats_df, use_container_width=True)

    # Plots
    st.subheader("📈 Graphiques")
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    dfp = strat.df
    r = res

    # Prix + stops
    ax = axes[0]
    ax.plot(dfp.index, dfp["close"], label="Close")
    ax.plot(dfp.index, r["stop"], label="Stop", linestyle="--")
    ax.plot(dfp.index, r["trailing"], label="Trailing", linestyle=":")
    ax.set_title("Prix & Stops")
    ax.legend()

    # Equity
    ax = axes[1]
    ax.plot(dfp.index, r["equity"], label="Equity")
    ax.set_title("Courbe d'équité")
    ax.legend()

    # Position & Leverage
    ax = axes[2]
    ax.plot(dfp.index, r["position"], label="Position")
    ax.plot(dfp.index, r["leverage"], label="Leverage", alpha=0.5)
    ax.set_title("Position & Levier")
    ax.legend()

    # Drawdown
    ax = axes[3]
    ax.plot(dfp.index, r["drawdown"], label="Drawdown")
    ax.axhline(-cfg.kill_switch_dd, color="red", linestyle="--", label="Kill-switch")
    ax.set_title("Drawdown")
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Note: les performances passées ne préjugent pas des performances futures.")
else:
    st.info("Réglez vos paramètres et cliquez sur **Lancer le backtest**.")
