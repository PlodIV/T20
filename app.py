import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("📈 ASX20 Daily & 7-Day Forecasting Tool")

ASX20 = {
    "BHP": "BHP.AX", "CSL": "CSL.AX", "CBA": "CBA.AX", "NAB": "NAB.AX",
    "WBC": "WBC.AX", "ANZ": "ANZ.AX", "WES": "WES.AX", "WOW": "WOW.AX",
    "TLS": "TLS.AX", "FMG": "FMG.AX", "MQG": "MQG.AX", "RIO": "RIO.AX",
    "WPL": "WPL.AX", "GMG": "GMG.AX", "BXB": "BXB.AX", "TCL": "TCL.AX",
    "SCG": "SCG.AX", "QAN": "QAN.AX", "SUN": "SUN.AX", "ALL": "ALL.AX"
}

@st.cache_data
def fetch_data(ticker):
    df = yf.download(ticker, start="2018-01-01", progress=False)
    df = df[['Close']].copy()
    df['Return'] = df['Close'].pct_change()
    df['7D_Return'] = df['Close'].pct_change(7)
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = MACD(close=df['Close']).macd_diff()
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df['Target_1D'] = (df['Return'].shift(-1) > 0).astype(int)
    df['Target_7D'] = (df['7D_Return'].shift(-7) > 0).astype(int)
    df.dropna(inplace=True)
    return df

def train_and_predict(df):
    features = ['RSI', 'MACD', 'Lag1', 'Lag2']
    X = df[features]
    y_1d = df['Target_1D']
    y_7d = df['Target_7D']

    X_train = X[:-7]
    y1_train = y_1d[:-7]
    y7_train = y_7d[:-7]

    X_test = X.iloc[[-1]]
    model1 = LogisticRegression(max_iter=1000).fit(X_train, y1_train)
    model7 = LogisticRegression(max_iter=1000).fit(X_train, y7_train)

    pred1 = model1.predict(X_test)[0]
    pred7 = model7.predict(X_test)[0]

    acc1 = accuracy_score(y1_train, model1.predict(X_train))
    acc7 = accuracy_score(y7_train, model7.predict(X_train))

    return pred1, acc1, pred7, acc7, df['Close'].iloc[-1]

results = []
for name, ticker in ASX20.items():
    try:
        df = fetch_data(ticker)
        pred1, acc1, pred7, acc7, price = train_and_predict(df)
        results.append({
            "Ticker": name,
            "Price": round(price, 2),
            "1D Forecast": "📈 Up" if pred1 else "📉 Down",
            "1D Accuracy": f"{acc1 * 100:.1f}%" if acc1 else "N/A",
            "7D Forecast": "📈 Up" if pred7 else "📉 Down",
            "7D Accuracy": f"{acc7 * 100:.1f}%" if acc7 else "N/A"
        })
    except Exception as e:
        results.append({
            "Ticker": name, "Price": "Error", "1D Forecast": "❌",
            "1D Accuracy": "-", "7D Forecast": "❌", "7D Accuracy": "-"
        })

df_results = pd.DataFrame(results)
st.dataframe(df_results, use_container_width=True)