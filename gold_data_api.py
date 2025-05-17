from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/gold-data')
def get_gold_data():
    try:
        df = yf.download('GC=F', start='2022-01-01', end='2024-01-01', auto_adjust=True)
        if df.empty:
            return jsonify({"error": "No data found."}), 404

        df = df.dropna().reset_index()
        return df.to_json(orient='records', date_format='iso')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
