from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/gold-data', methods=['GET'])
def get_gold_data():
    df = yf.download("GC=F", start="2022-01-01", end="2024-01-01")
    df.reset_index(inplace=True)  # Ensures 'Date' is a column
    return df.to_json(orient="records")  # return as list of dicts

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
