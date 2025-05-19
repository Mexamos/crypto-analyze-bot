import os
from datetime import datetime, timedelta
from typing import List

import san
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from app.config import Config
from app.crypto.santimentapi_client import SantimentApiClient, NoDataReturned, InvalidMetricResponse


class ModelTrainingFacade:

    def __init__(
        self, api_client: SantimentApiClient, config: Config
    ):
        self.api_client = api_client
        self.config = config
        self.model_file_path = config.santimentapi_model_file_path
        self.scaler_file_path = config.santimentapi_scaler_file_path
        self.date_format = config.santimentapi_date_format

    def train_and_save_model(
        self, slugs: List[str], from_date: str, to_date=None, interval="1d", label_threshold=0.01
    ):
        try:
            datetime.strptime(from_date, self.date_format)
        except ValueError:
            raise ValueError(f"'{from_date}' is not a valid date in {self.date_format} format.")
        end_date = to_date or datetime.now().strftime(self.date_format)

        df = self._prepare_dataset(slugs, from_date, end_date, interval, label_threshold)
        X = df.drop(columns=["label"])
        y = df["label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        classification_report(y_test, y_pred)

        dump(clf, self.model_file_path)
        dump(scaler, self.scaler_file_path)

    def _prepare_dataset(self, slugs, from_date, to_date, interval, label_threshold):
        frames = []
        metrics = [
            "price_usd",
            "active_addresses_24h",
            "active_addresses_24h_change_1d",
            "active_addresses_24h_change_30d",
            "30d_moving_avg_dev_activity_change_1d",
        ]
        for slug in slugs:
            data = {}
            for metric in metrics:
                try:
                    ts = self.api_client.fetch_timeseries(slug, metric, from_date, to_date, interval)
                    data[metric] = ts
                except (NoDataReturned, InvalidMetricResponse):
                    # Skip this metric if data is missing or invalid
                    continue

            # Need at least one metric to build DataFrame
            if not data:
                continue

            # Merge available metrics
            metrics_available = list(data.keys())
            df_agg: pd.DataFrame = data[metrics_available[0]]
            for m in metrics_available[1:]:
                df_agg = df_agg.join(data[m][m], how="inner")

            df_agg.reset_index(inplace=True)
            df_agg["slug"] = slug
            frames.append(self._label_price_movement(df_agg, label_threshold))

        full_df = pd.concat(frames, ignore_index=True)
        return full_df.drop(columns=["price_usd", "price_next", "change", "datetime", "slug"])

    def _label_price_movement(self, df: pd.DataFrame, threshold):
        df = df.copy()
        df["price_next"] = df["price_usd"].shift(-1)
        df["change"] = (df["price_next"] - df["price_usd"]) / df["price_usd"]
        df["label"] = df["change"].apply(
            lambda x: "UP" if x > threshold else ("DOWN" if x < -threshold else "STABLE")
        )
        return df.dropna()


class ModelPredictionFacade:

    def __init__(
        self, api_client: SantimentApiClient, config: Config
    ):
        self.api_client = api_client
        self.config = config
        self.model_file_path = config.santimentapi_model_file_path
        self.scaler_file_path = config.santimentapi_scaler_file_path
        self.date_format = config.santimentapi_date_format

    def predict(self, slugs: List[str], days_back=2, interval="1d"):
        model: RandomForestClassifier = load(self.model_file_path)
        scaler: StandardScaler = load(self.scaler_file_path)

        today = datetime.now().date()
        from_date = (today - timedelta(days=days_back)).strftime(self.date_format)
        to_date = today.strftime(self.date_format)
        metrics = [
            "active_addresses_24h",
            "active_addresses_24h_change_1d",
            "active_addresses_24h_change_30d",
            "30d_moving_avg_dev_activity_change_1d",
        ]

        lines = [f"🔮 Prediction for {(today + timedelta(days=1)):%B %d, %Y}"]
        for slug in slugs:
            try:
                features = {}
                for metric in metrics:
                    try:
                        df = self.api_client.fetch_timeseries(slug, metric, from_date, to_date, interval)
                        features[metric] = df.iloc[-1][metric]
                    except (NoDataReturned, InvalidMetricResponse):
                        # Skip missing/invalid metric
                        continue

                if not features:
                    lines.append(f"{slug} → ⚠️ Failed to fetch any metrics")
                    continue

                # Align features with scaler's expected input
                cols = list(scaler.feature_names_in_)
                features_df = pd.DataFrame([features]).reindex(columns=cols, fill_value=0)
                X_scaled = scaler.transform(features_df)
                pred = model.predict(X_scaled)[0]
                signal_emoji = {"UP": "🔼", "DOWN": "🔽", "STABLE": "⏸️"}[pred]
                lines.append(f"{slug.capitalize()} → {signal_emoji} Likely to go {pred}")
            except Exception:
                lines.append(f"{slug} → ⚠️ Failed to predict")

        return "\n".join(lines)
