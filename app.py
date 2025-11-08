from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder="static")

# ====== モデル読み込み ======
model = joblib.load("model/hasc_lgbm.pkl")
scaler = joblib.load("model/scaler.pkl")

# ====== ルート（index.html） ======
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ====== 行動分類API ======
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # JSONから加速度データを取得
    acc_x = np.array(data["AccX"])
    acc_y = np.array(data["AccY"])
    acc_z = np.array(data["AccZ"])

    # ====== 特徴量抽出 ======
    features = []
    for axis in [acc_x, acc_y, acc_z]:
        features.extend([
            np.mean(axis),
            np.std(axis),
            np.max(axis),
            np.min(axis),
            np.median(axis)
        ])

    # スケーリング & 予測
    X = scaler.transform([features])
    pred = model.predict(X)[0]

    labels = ["stay", "walk", "skip"]
    return jsonify({"prediction": labels[pred]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
