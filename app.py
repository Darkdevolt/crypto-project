import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Solana Trading Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Fonctions d'analyse technique
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, slow=26, fast=12, signal=9):
    data['EMA_fast'] = data['Close'].ewm(span=fast).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['Signal'] = data['MACD'].ewm(span=signal).mean()
    return data

def calculate_bollinger_bands(data, window=20):
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper'] = data['MA'] + (data['STD'] * 2)
    data['Lower'] = data['MA'] - (data['STD'] * 2)
    return data

# Fonction de backtest
def backtest_strategy(data):
    # Initialisation des variables
    capital = 10000
    position = 0
    buy_price = 0
    trades = []
    in_position = False
    
    # Conditions de trading
    for i in range(1, len(data)):
        # Conditions d'achat
        buy_condition = (
            data['RSI'][i] < 45 and
            data['Close'][i] > data['MA50'][i] and
            data['MA50'][i] > data['MA200'][i] and
            not in_position
        )
        
        # Conditions de vente
        sell_condition = (
            (data['RSI'][i] > 70 or 
             data['Close'][i] < data['MA50'][i]) and 
            in_position
        )
        
        # Exécution des trades
        if buy_condition:
            position = capital / data['Close'][i]
            buy_price = data['Close'][i]
            trades.append(('buy', data.index[i], data['Close'][i], capital))
            in_position = True
            
        elif sell_condition and in_position:
            capital = position * data['Close'][i]
            profit = ((data['Close'][i] - buy_price) / buy_price) * 100
            trades.append(('sell', data.index[i], data['Close'][i], capital, profit))
            position = 0
            in_position = False
    
    # Fermer la position à la fin si nécessaire
    if in_position:
        capital = position * data['Close'].iloc[-1]
        profit = ((data['Close'].iloc[-1] - buy_price) / buy_price) * 100
        trades.append(('sell', data.index[-1], data['Close'].iloc[-1], capital, profit))
    
    return trades, capital

# Interface Streamlit
st.title('🚀 Solana Trading Dashboard')
st.markdown("Analyse technique et signaux de trading pour Solana (SOL-USD)")

# Sélecteur de période
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Date de début", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("Date de fin", datetime.now())

# Téléchargement des données
@st.cache_data
def load_data():
    return yf.download('SOL-USD', start=start_date, end=end_date)

data = load_data()

if data.empty:
    st.error("Erreur: Aucune donnée disponible pour cette période.")
    st.stop()

# Calcul des indicateurs
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['RSI'] = calculate_rsi(data)
data = calculate_macd(data)
data = calculate_bollinger_bands(data)

# Dernières valeurs
last_close = data['Close'].iloc[-1]
last_rsi = data['RSI'].iloc[-1]
ma50_diff = (last_close - data['MA50'].iloc[-1]) / data['MA50'].iloc[-1] * 100
ma200_diff = (last_close - data['MA200'].iloc[-1]) / data['MA200'].iloc[-1] * 100

# Signal de trading
signal = "Attendre"
signal_color = "gray"
recommendation = "Maintenir une position neutre"

buy_condition = (
    last_rsi < 45 and
    last_close > data['MA50'].iloc[-1] and
    data['MA50'].iloc[-1] > data['MA200'].iloc[-1]
)

sell_condition = (
    last_rsi > 70 or 
    last_close < data['MA50'].iloc[-1]
)

if buy_condition:
    signal = "ACHETER"
    signal_color = "green"
    recommendation = "Conditions favorables pour une position longue"
elif sell_condition:
    signal = "VENDRE"
    signal_color = "red"
    recommendation = "Prendre des profits ou réduire l'exposition"

# Afficher les KPIs
st.subheader(f"Dernier cours: ${last_close:.2f}")
st.markdown(f"### Signal actuel: :{signal_color}[{signal}]")
st.caption(recommendation)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("RSI (14)", f"{last_rsi:.2f}", "Survendu" if last_rsi < 30 else "Suracheté" if last_rsi > 70 else "Neutre")
kpi2.metric("Diff MA50", f"{ma50_diff:.2f}%", "Au-dessus" if ma50_diff > 0 else "En-dessous")
kpi3.metric("Diff MA200", f"{ma200_diff:.2f}%", "Au-dessus" if ma200_diff > 0 else "En-dessous")
kpi4.metric("Volatilité (30j)", f"{data['Close'].pct_change().std()*100:.2f}%")

# Graphiques
tab1, tab2, tab3 = st.tabs(["Graphique des prix", "Indicateurs techniques", "Backtest"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       row_heights=[0.7, 0.3])
    
    # Prix et moyennes mobiles
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Prix'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], 
                             line=dict(color='blue', width=1.5), 
                             name='MA 50'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], 
                             line=dict(color='orange', width=1.5), 
                             name='MA 200'), row=1, col=1)
    
    # Bandes de Bollinger
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], 
                             line=dict(color='rgba(255,255,255,0)'), 
                             showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], 
                             line=dict(color='rgba(255,255,255,0)'),
                             fill='tonexty',
                             fillcolor='rgba(100, 100, 255, 0.2)',
                             name='Bollinger Bands'), row=1, col=1)
    
    # Volume
    colors = ['green' if data['Close'][i] > data['Open'][i] else 'red' 
             for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], 
                         marker_color=colors, 
                         name='Volume'), row=2, col=1)
    
    fig.update_layout(
        title='Solana (SOL-USD) - Analyse des prix',
        height=700,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                     line=dict(color='cyan'), 
                                     name='RSI'))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.update_layout(
            title='RSI (14 jours)',
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
        
    with col2:
        # Graphique MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Bar(x=data.index, y=data['MACD'], 
                                 name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data['Signal'], 
                                     line=dict(color='orange'), 
                                     name='Signal'))
        fig_macd.update_layout(
            title='MACD (12, 26, 9)',
            height=400,
            template='plotly_dark'
        )
        st.plotly_chart(fig_macd, use_container_width=True)

with tab3:
    st.subheader("Backtest de la stratégie de trading")
    
    # Exécuter le backtest
    trades, final_capital = backtest_strategy(data)
    initial_capital = 10000
    profit = final_capital - initial_capital
    profit_pct = (profit / initial_capital) * 100
    
    # Performance
    buy_hold = (data['Close'].iloc[-1] / data['Close'].iloc[0]) * initial_capital
    buy_hold_profit = buy_hold - initial_capital
    buy_hold_pct = (buy_hold_profit / initial_capital) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Capital final", f"${final_capital:,.2f}", 
                f"{profit_pct:.2f}% de profit")
    col2.metric("Buy & Hold", f"${buy_hold:,.2f}", 
                f"{buy_hold_pct:.2f}% de profit")
    
    # Graphique de performance
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                 name='Prix Solana', 
                                 line=dict(color='rgba(100, 100, 255, 0.5)')))
    
    # Ajouter les trades
    buy_dates = [trade[1] for trade in trades if trade[0] == 'buy']
    buy_prices = [trade[2] for trade in trades if trade[0] == 'buy']
    sell_dates = [trade[1] for trade in trades if trade[0] == 'sell']
    sell_prices = [trade[2] for trade in trades if trade[0] == 'sell']
    
    fig_perf.add_trace(go.Scatter(x=buy_dates, y=buy_prices, 
                                 mode='markers', 
                                 marker=dict(size=10, color='green'),
                                 name='Achat'))
    
    fig_perf.add_trace(go.Scatter(x=sell_dates, y=sell_prices, 
                                 mode='markers', 
                                 marker=dict(size=10, color='red'),
                                 name='Vente'))
    
    # Relier les trades
    for i in range(len(buy_dates)):
        fig_perf.add_trace(go.Scatter(
            x=[buy_dates[i], sell_dates[i]],
            y=[buy_prices[i], sell_prices[i]],
            mode='lines',
            line=dict(color='yellow', dash='dash'),
            showlegend=False
        ))
    
    fig_perf.update_layout(
        title='Performance de la stratégie',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Détails des trades
    if trades:
        st.subheader("Détails des trades")
        trade_df = pd.DataFrame([{
            'Type': trade[0],
            'Date': trade[1],
            'Prix': trade[2],
            'Capital': f"${trade[3]:,.2f}",
            'Profit (%)': trade[4] if len(trade) > 4 else None
        } for trade in trades])
        st.dataframe(trade_df.style.format({'Profit (%)': '{:.2f}%'}), height=300)
    else:
        st.info("Aucun trade exécuté pendant cette période")

# Analyse de marché
st.subheader("📈 Analyse de marché actuelle")
st.markdown(f"""
**Perspective technique:**
- Le RSI actuel à **{last_rsi:.2f}** indique un marché **{"survendu" if last_rsi < 30 else "suracheté" if last_rsi > 70 else "neutre"}**.
- Le prix est **{"au-dessus" if last_close > data['MA50'].iloc[-1] else "en-dessous"}** de la moyenne mobile 50 jours.
- La tendance à moyen terme est **{"haussière" if data['MA50'].iloc[-1] > data['MA200'].iloc[-1] else "baissière"}**.

**Recommandation:**
- **{signal}**: {recommendation}
- Les prochains niveaux clés à surveiller:
  - Résistance: ${data['Upper'].iloc[-1]:.2f}
  - Support: ${data['Lower'].iloc[-1]:.2f}
""")

# Footer
st.divider()
st.caption("""
**Disclaimer:** Cette application fournit des analyses techniques et des signaux de trading à titre informatif uniquement. 
Ne constitue pas un conseil en investissement. Les crypto-monnaies sont volatiles - investissez uniquement ce que vous pouvez vous permettre de perdre.
Données fournies par Yahoo Finance. Mise à jour quotidienne.
""")
