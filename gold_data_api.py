from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/gold-data', methods=['GET'])
def get_gold_data():
    df = yf.download("GC=F", start="2022-01-01", end="2024-01-01")
    df.reset_index(inplace=True)

    # 🛠 Flatten the column index if it's multi-level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.to_json(orient="records")
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

