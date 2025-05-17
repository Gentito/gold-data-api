from flask import Flask, jsonify
import yfinance as yf

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Gold Data API!"})

@app.route('/gold-data', methods=['GET'])
def get_gold_data():
    try:
        # Fetch gold futures data from Yahoo Finance
        df = yf.download("GC=F", start="2022-01-01", end="2024-01-01")
        if 'Date' not in df.columns:
            df.reset_index(inplace=True) # Promote index to column
            df.rename(columns={"Date": "Date"}, inplace=True)  # Optional no-op
        return df.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

