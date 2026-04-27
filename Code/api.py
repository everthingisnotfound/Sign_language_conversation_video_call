from flask import Flask, jsonify
import threading
import time

app = Flask(__name__)

latest_prediction = "..."

def run_model():
    global latest_prediction
    while True:
        # 🔥 CALL YOUR EXISTING LOGIC HERE
        # Instead of print, store result

        latest_prediction = "Detected Gesture"  # replace later
        time.sleep(1)

# run model in background
threading.Thread(target=run_model, daemon=True).start()

@app.route("/predict", methods=["GET"])
def predict():
    return jsonify({"text": latest_prediction})

if __name__ == "__main__":
    app.run(port=5000)