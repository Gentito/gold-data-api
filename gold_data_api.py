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
        df.reset_index(inplace=True)  # Ensure 'Date' becomes a column
        return df.to_json(orient="records")  # List of dicts
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

