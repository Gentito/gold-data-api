import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import backtrader as bt
from ta.trend import MACD

# 1. Fetch Gold price data from Yahoo Finance
def fetch_xauusd_yfinance(from_date='2022-01-01', to_date='2024-01-01'):
    for symbol in ['XAUUSD=X', 'GC=F']:
        print(f"📡 Attempting to download: {symbol}")
        df = yf.download(symbol, start=from_date, end=to_date, auto_adjust=True)
        
        if not df.empty:
            print(f"✅ Successfully downloaded: {symbol}")
            
            # Make sure 'Volume' exists and is valid
            if 'Volume' not in df.columns or df['Volume'].isna().all():
                print("⚠️ Volume data missing — filling with zeros.")
                df['Volume'] = 0.0

            df = df.dropna()

            try:
                df = add_all_ta_features(
                    df, open='Open', high='High', low='Low',
                    close='Close', volume='Volume', fillna=True
                )
            except Exception as e:
                print(f"⚠️ TA library failed: {e}. Using MACD only.")
                macd = MACD(close=df['Close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()

            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            return df

        print(f"❌ Failed to download: {symbol}")

    raise ValueError("❗ No valid data found for XAUUSD=X or GC=F.")


# 2. Prepare input sequences for LSTM
def prepare_features(df, seq_len=30):
    df = df.copy()
    X = df.select_dtypes(include=[np.number]).drop(['Target'], axis=1)
    y = df['Target'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq), scaler

# 3. LSTM Model Definition
def create_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Backtrader Strategy
class LSTMStrategy(bt.Strategy):
    def __init__(self, model, scaler, lookback=30):
        self.model = model
        self.scaler = scaler
        self.lookback = lookback
        self.dataX = []

    def next(self):
        self.dataX.append([
            self.data.open[0],
            self.data.high[0],
            self.data.low[0],
            self.data.close[0],
            self.data.volume[0],
        ])

        if len(self.dataX) < self.lookback:
            return
        if len(self.dataX) > self.lookback:
            self.dataX.pop(0)

        input_scaled = self.scaler.transform(self.dataX)
        prediction = self.model.predict(np.array([input_scaled]), verbose=0)[0][0]

        if prediction > 0.6 and not self.position:
            self.buy()
        elif prediction < 0.4 and self.position:
            self.close()

# 5. Run Everything
def run_trading_bot():
    print("🚀 Starting trading bot...")
    df = fetch_xauusd_yfinance(from_date='2022-01-01', to_date='2024-01-01')

    if df.empty:
        print("❌ DataFrame is empty. Exiting.")
        return

    print(f"📈 Data loaded with {len(df)} rows.")
    X, y, scaler = prepare_features(df)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("🧠 Training LSTM...")
    model = create_lstm(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    df_bt = df.iloc[split + 30:].copy().reset_index()
    df_bt['OpenInterest'] = 0

    class PandasData(bt.feeds.PandasData):
        lines = ('openinterest',)
        params = (
            ('datetime', 'Date'),
            ('open', 'Open'),
            ('high', 'High'),
            ('low', 'Low'),
            ('close', 'Close'),
            ('volume', 'Volume'),
            ('openinterest', 'OpenInterest'),
        )

    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addstrategy(LSTMStrategy, model=model, scaler=scaler)
    data_feed = PandasData(dataname=df_bt)
    cerebro.adddata(data_feed)

    print("💰 Initial portfolio value:", cerebro.broker.getvalue())
    cerebro.run()
    print("📈 Final portfolio value:", cerebro.broker.getvalue())
    cerebro.plot()

# Run the bot
if __name__ == "__main__":
    run_trading_bot()
