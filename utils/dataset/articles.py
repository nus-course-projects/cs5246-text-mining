import os
import sqlite3
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd
import matplotlib.pyplot as plt


class ArticlesDataset:
    def __init__(self, data_dir: str, filtered_rows: Optional[pd.DataFrame] = None, filtered_labels: Optional[pd.DataFrame] = None) -> None:
        self.data_dir = data_dir
        self.bin_file_handle = open(os.path.join(data_dir, "news_data.bin"), "rb")
        self.conn = sqlite3.connect(os.path.join(data_dir, "index.db"))
        self.label_conn = sqlite3.connect(os.path.join(data_dir, "labels.db"))

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.label_cursor = self.label_conn.cursor()
        if filtered_rows is None:
            self.filtered_rows = self._load_all_rows()
            self.filtered_labels = self._load_all_labels()
        else:
            self.filtered_rows = filtered_rows
            self.filtered_labels = filtered_labels

        assert len(self.filtered_rows) == len(self.filtered_labels), "Filtered rows and labels must have the same length"

        self._utc = ZoneInfo("UTC")

    def __del__(self):
        self.bin_file_handle.close()
        self.conn.close()
        self.label_conn.close()

    def _load_all_rows(self):
        query = "SELECT * FROM articles"
        return pd.read_sql_query(query, self.conn)

    def _load_all_labels(self):
        query = "SELECT * FROM labels"
        return pd.read_sql_query(query, self.label_conn)

    def __getitem__(self, idx: int):
        row = self.filtered_rows.iloc[idx]

        metadata = {
            "title": row["title"],
            "url": row["url"],
            "date": datetime.fromtimestamp(row["date"], tz=self._utc),
            "country": row["country"],
            "query": row["query"]
        }

        self.bin_file_handle.seek(row["offset"])
        content = self.bin_file_handle.read(row["length"]).decode("utf-8")

        label_row = self.filtered_labels.iloc[idx]
        label = {
            "event_occured": (label_row["event_occured"] == 1).item(),
            "event": label_row["event"],
            "impact": int(label_row["impact"]) if not pd.isna(label_row["impact"]) else None,
            "dt": datetime.strptime(label_row["dt"], "%Y-%m-%d").replace(tzinfo=self._utc) if label_row["dt"] is not None else None,
            "loc": label_row["loc"],
            "city": label_row["city"],
            "state": label_row["state"],
            "country": label_row["country"],
            "latitude": float(label_row["latitude"]) if not pd.isna(label_row["latitude"]) else None,
            "longitude": float(label_row["longitude"]) if not pd.isna(label_row["longitude"]) else None
        }

        return metadata, content, label

    def __len__(self):
        return len(self.filtered_rows)

    def filter_by_metadata(self, queries: list[str] = [], dates: Optional[list[datetime] | tuple[datetime, datetime]] = None):
        conditions: list[str] = []
        parameters: list[str | float] = []

        if queries:
            conditions.append("query IN ({})".format(",".join("?" * len(queries))))
            parameters.extend(queries)

        if dates:
            if isinstance(dates, tuple) and len(dates) == 2:
                conditions.append("date BETWEEN ? AND ?")
                parameters.append(dates[0].timestamp())
                parameters.append(dates[1].timestamp())
            else:
                # Handle list of dates
                conditions.append("date IN ({})".format(",".join("?" * len(dates))))
                parameters.extend([date.timestamp() if isinstance(date, datetime) else date for date in dates])

        query_str = "SELECT * FROM articles"
        if conditions:
            query_str += " WHERE " + " AND ".join(conditions)

        filtered_data = pd.read_sql_query(query_str, self.conn, params=parameters)  # type: ignore
        if filtered_data.empty:
            raise ValueError("No matching rows found")
        index_ids = filtered_data['index_id'].tolist()
        label_query = "SELECT * FROM labels WHERE index_id IN ({})".format(",".join("?" * len(index_ids)))
        filtered_labels = pd.read_sql_query(label_query, self.label_conn, params=index_ids)

        assert filtered_data.shape[0] == filtered_labels.shape[0], "Filtered rows and labels must have the same length"
        return ArticlesDataset(self.data_dir, filtered_data, filtered_labels)

    def filter_by_label(
        self,
        event_occured: Optional[bool] = None,
        events: Optional[list[str]] = None,
        impacts: Optional[list[int]] = None,
        city: Optional[list[str]] = None,
        state: Optional[list[str]] = None,
        country: Optional[list[str]] = None,
        city_na: Optional[bool] = None,
        state_na: Optional[bool] = None,
        country_na: Optional[bool] = None
    ):
        conditions: list[str] = []
        parameters: list[str | int] = []

        if event_occured is not None:
            conditions.append("event_occured = ?")
            parameters.append(int(event_occured))

        if events:
            conditions.append("event IN ({})".format(",".join("?" * len(events))))
            parameters.extend(events)

        if impacts:
            conditions.append("impact IN ({})".format(",".join("?" * len(impacts))))
            parameters.extend(impacts)

        if city:
            conditions.append("city IN ({})".format(",".join("?" * len(city))))
            parameters.extend(city)

        if state:
            conditions.append("state IN ({})".format(",".join("?" * len(state))))
            parameters.extend(state)

        if country:
            conditions.append("country IN ({})".format(",".join("?" * len(country))))
            parameters.extend(country)

        if city_na:
            conditions.append("city IS NULL")

        if state_na:
            conditions.append("state IS NULL")

        if country_na:
            conditions.append("country IS NULL")

        query_str = "SELECT * FROM labels"
        if conditions:
            query_str += " WHERE " + " AND ".join(conditions)

        filtered_labels = pd.read_sql_query(query_str, self.label_conn, params=parameters)  # type: ignore
        if filtered_labels.empty:
            raise ValueError("No matching rows found")

        index_ids = filtered_labels['index_id'].tolist()
        article_query = "SELECT * FROM articles WHERE index_id IN ({})".format(",".join("?" * len(index_ids)))
        filtered_data = pd.read_sql_query(article_query, self.conn, params=index_ids)
        assert filtered_data.shape[0] == filtered_labels.shape[0], "Filtered rows and labels must have the same length"

        return ArticlesDataset(self.data_dir, filtered_data, filtered_labels)

    def show_dist(self, plot: bool = False) -> None:
        query_dist = self.filtered_rows['query'].value_counts()
        date_dist = self.filtered_rows['date'].value_counts().sort_index()

        print(query_dist)
        print(date_dist)

        event_occured_dist = self.filtered_labels['event_occured'].value_counts()
        events_dist = self.filtered_labels['event'].value_counts()
        impact_dist = self.filtered_labels['impact'].value_counts()
        city_dist = self.filtered_labels['city'].value_counts()
        state_dist = self.filtered_labels['state'].value_counts()
        country_dist = self.filtered_labels['country'].value_counts()

        print(event_occured_dist)
        print(events_dist)
        print(impact_dist)
        print(city_dist)
        print(state_dist)
        print(country_dist)

        if plot:
            query_dist.plot(kind='bar')
            plt.show()
            event_occured_dist.plot(kind='bar')
            plt.show()
            events_dist.plot(kind='bar')
            plt.show()
            impact_dist.plot(kind='bar')
            plt.show()
