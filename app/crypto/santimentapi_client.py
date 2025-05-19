from datetime import datetime, timedelta
from typing import List

import san
import pandas as pd


class SantimentApiClientException(Exception):
    pass


class NoDataReturned(SantimentApiClientException):
    pass


class InvalidMetricResponse(SantimentApiClientException):
    pass


class InvalidSlug(SantimentApiClientException):
    pass


class SantimentApiClient:

    def __init__(self, api_key: str):
        self.api_key = api_key
        san.ApiConfig.api_key = api_key

        self._projects_df = None

    def get_all_projects(self) -> pd.DataFrame:
        if self._projects_df is None:
            df = san.get("projects/all")
            if df is None or df.empty:
                raise NoDataReturned("No projects data returned from Santiment API")

            self._projects_df = df

        return self._projects_df

    def validate_slugs(self, slugs: List[str]) -> None:
        df = self.get_all_projects()
        valid_slugs = set(df["slug"].tolist())
        invalid = [slug for slug in slugs if slug not in valid_slugs]
        if invalid:
            raise InvalidSlug(f"Invalid slugs: {', '.join(invalid)}")

    def emerging_trends(self, from_date: str, to_date: str, interval: str = "1h", size: int = 10):
        trends = san.get(
            "emerging_trends",
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            size=size
        )
        return trends["word"].unique()

    def fetch_timeseries(
        self, slug: str, metric: str, from_date: str, to_date: str, interval: str
    ) -> pd.DataFrame:
        try:
            df = san.get(metric, slug=slug, from_date=from_date, to_date=to_date, interval=interval)
        except Exception as exc:
            raise SantimentApiClientException(f"Error fetching {metric} for {slug}: {exc}")

        if df is None or df.empty:
            raise NoDataReturned(
                f"No data returned for metric '{metric}' and slug '{slug}' between {from_date} and {to_date}'"
            )
        if 'value' not in df.columns:
            raise InvalidMetricResponse(
                f"Expected 'value' column not found in response for metric '{metric}' and slug '{slug}'"
            )

        return df.rename(columns={"value": metric})

    def fetch_latest_metrics(
        self, slug: str, metrics: list, from_date: str, to_date: str, interval: str
    ) -> pd.Series:
        features = {}
        for metric in metrics:
            df = self.fetch_timeseries(slug, metric, from_date, to_date, interval)
            features[metric] = df.iloc[-1][metric]

        return pd.Series(features)
