import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ======== ① データのルートパスを指定 ========
DATA_ROOT = "./HASC_dataset"  # ← あなたの実際のデータパスに合わせる

# ======== ② フォルダ名とラベルの対応 ========
LABELS = {
    "1_stay": 0,
    "2_walk": 1,
    "4_skip": 2
}

# ======== ③ CSVファイルの読み込み関数 ========
def load_csv_files():
    X, y = [], []

    for label_name, label_id in LABELS.items():
        folder_path = os.path.join(DATA_ROOT, label_name)
        persons = glob.glob(os.path.join(folder_path, "person*"))
        for person_folder in persons:
            csv_files = glob.glob(os.path.join(person_folder, "*.csv"))
            for csv_file in csv_files:
                try:
                    # ヘッダーがないので明示的に指定
                    df = pd.read_csv(csv_file, header=None, names=["Time", "AccX", "AccY", "AccZ"])

                    # データ行があるか確認
                    if df.shape[0] < 10:
                        print(f"⚠️ Skipped (too short): {csv_file}")
                        continue

                    # 特徴量抽出
                    features = extract_features(df)
                    X.append(features)
                    y.append(label_id)

                except Exception as e:
                    print(f"❌ Error reading {csv_file}: {e}")

    return np.array(X), np.array(y)


# ======== ④ 特徴量抽出（シンプル統計特徴） ========
def extract_features(df):
    feats = []
    for axis in ["AccX", "AccY", "AccZ"]:
        feats.extend([
            df[axis].mean(),
            df[axis].std(),
            df[axis].max(),
            df[axis].min(),
            df[axis].median(),
        ])
    return feats

# ======== ⑤ メイン処理 ========
def main():
    print("📂 Loading data...")
    X, y = load_csv_files()
    print(f"✅ Loaded: {len(X)} samples")

    # ======== データ分割 ========
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ======== 特徴量スケーリング ========
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ======== モデル学習 ========
    print("🚀 Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ======== 評価 ========
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["stay", "walk", "skip"]))

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # ======== モデル保存 ========
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/hasc_lgbm.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("💾 Saved model to model/hasc_lgbm.pkl")

if __name__ == "__main__":
    main()
