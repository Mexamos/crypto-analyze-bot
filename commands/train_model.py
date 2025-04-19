
import argparse
import os
import sys
from datetime import datetime

import san
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--from_date", required=True, help="Start date for fetching data (YYYY-MM-DD)")
args = parser.parse_args()

try:
    datetime.strptime(args.from_date, "%Y-%m-%d")
except ValueError:
    print(f"❌ Error: '{args.from_date}' is not a valid date in YYYY-MM-DD format.")
    sys.exit(1)

SANTIMENT_API_KEY = os.getenv('SANTIMENT_API_KEY')
san.ApiConfig.api_key = SANTIMENT_API_KEY

SLUGS = ["bitcoin", "ethereum"]
FROM_DATE = args.from_date
TO_DATE = datetime.utcnow().strftime("%Y-%m-%d")
INTERVAL = "1d"
LABEL_THRESHOLD = 0.01  # 1%

METRICS = [
    "price_usd",
    "active_addresses_24h",
    "active_addresses_24h_change_1d",
    "active_addresses_24h_change_30d",
    "30d_moving_avg_dev_activity_change_1d"
]


def fetch_data(slug):
    data = {}
    for metric in METRICS:
        df = san.get(metric, slug=slug, from_date=FROM_DATE, to_date=TO_DATE, interval=INTERVAL)
        df = df.rename(columns={"value": metric})
        data[metric] = df

    # Merge all metrics by datetime
    df_final = data[METRICS[0]]
    for metric in METRICS[1:]:
        df_final = df_final.join(data[metric][metric], how="inner")

    df_final.reset_index(inplace=True)
    df_final["slug"] = slug
    return df_final


def label_price_movement(df):
    df["price_next"] = df["price_usd"].shift(-1)
    df["change"] = (df["price_next"] - df["price_usd"]) / df["price_usd"]
    df["label"] = df["change"].apply(
        lambda x: "UP" if x > LABEL_THRESHOLD else ("DOWN" if x < -LABEL_THRESHOLD else "STABLE")
    )
    return df.dropna()


def prepare_dataset():
    frames = []
    for slug in SLUGS:
        df = fetch_data(slug)
        df = label_price_movement(df)
        frames.append(df)

    full_df = pd.concat(frames)
    full_df = full_df.drop(columns=["price_usd", "price_next", "change", "datetime", "slug"])
    return full_df


def train_and_save_model():
    df = prepare_dataset()
    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    dump(clf, "model.pkl")
    dump(scaler, "scaler.pkl")


if __name__ == "__main__":
    train_and_save_model()
