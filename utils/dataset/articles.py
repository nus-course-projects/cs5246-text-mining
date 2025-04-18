import os
import sqlite3
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd
import matplotlib.pyplot as plt


class ArticlesDataset:
    def __init__(self, data_dir: str, filtered_rows: Optional[pd.DataFrame] = None, filtered_labels: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the ArticlesDataset class.

        Args:
            data_dir (str): The directory containing the dataset files.
            filtered_rows (Optional[pd.DataFrame]): Optional pre-filtered DataFrame for article rows. If None, all rows are loaded.
            filtered_labels (Optional[pd.DataFrame]): Optional pre-filtered DataFrame for article labels. If None, all labels are loaded.

        Attributes:
            data_dir (str): Directory where data files are stored.
            bin_file_handle (file object): File handle for reading binary news data.
            conn (sqlite3.Connection): SQLite connection for the article index database.
            label_conn (sqlite3.Connection): SQLite connection for the labels database.
            cursor (sqlite3.Cursor): Cursor for executing queries on the article index database.
            label_cursor (sqlite3.Cursor): Cursor for executing queries on the labels database.
            filtered_rows (pd.DataFrame): DataFrame containing article rows.
            filtered_labels (pd.DataFrame): DataFrame containing article labels.
            _utc (ZoneInfo): Time zone information for UTC.
        """
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
        """
        Close all open file handles and database connections when the object is
        garbage collected.
        """
        self.bin_file_handle.close()
        self.conn.close()
        self.label_conn.close()

    def _load_all_rows(self):
        """
        Load all rows from the SQLite database into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all article rows.
        """
        query = "SELECT * FROM articles"
        return pd.read_sql_query(query, self.conn)

    def _load_all_labels(self):
        """
        Load all labels from the SQLite database into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all article labels.
        """
        query = "SELECT * FROM labels"
        return pd.read_sql_query(query, self.label_conn)

    def __getitem__(self, idx: int):
        """
        Retrieve the metadata, content, and label for an article at a specific index.

        This method allows access to the dataset by providing an index. It returns
        a tuple containing the metadata, content, and label information for the
        article located at the given index.

        Args:
            idx (int): The index of the article to retrieve.

        Returns:
            tuple: A tuple containing:
                - metadata (dict): A dictionary with keys 'title', 'url', 'date',
                'country', and 'query' representing the article's metadata.
                - content (str): The full content of the article.
                - label (dict): A dictionary representing the label information
                with keys such as 'event_occured', 'event', 'impact', 'dt', 'loc',
                'city', 'state', 'country', 'latitude', 'longitude'.
        """
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
        """
        Get the number of articles in the dataset.

        Returns:
            int: The number of articles in the dataset.
        """
        return len(self.filtered_rows)

    def filter_by_metadata(self, queries: list[str] = [], dates: Optional[list[datetime] | tuple[datetime, datetime]] = None):
        """
        Filter the dataset by metadata criteria.

        Filter the dataset by providing a list of query strings and/or a tuple or list
        of dates. The filtered dataset will contain only those articles whose metadata
        matches the provided criteria.

        Args:
            queries (list[str], optional): A list of query strings to match. Defaults to [].
            dates (list[datetime] | tuple[datetime, datetime], optional): A list or tuple of
                datetime objects representing the date range to filter by. Defaults to None.

        Returns:
            ArticlesDataset: A new dataset containing only the articles that match the
                provided criteria.
        """
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
        """
        Filter the dataset by label criteria.

        Filter the dataset by providing criteria for the label fields. The filtered
        dataset will contain only those articles whose labels match the provided
        criteria.

        Args:
            event_occured (bool, optional): Filter by whether an event occurred. Defaults to None.
            events (list[str], optional): Filter by a list of event names. Defaults to None.
            impacts (list[int], optional): Filter by a list of impact values. Defaults to None.
            city (list[str], optional): Filter by a list of city names. Defaults to None.
            state (list[str], optional): Filter by a list of state names. Defaults to None.
            country (list[str], optional): Filter by a list of country names. Defaults to None.
            city_na (bool, optional): Filter by whether city is NULL. Defaults to None.
            state_na (bool, optional): Filter by whether state is NULL. Defaults to None.
            country_na (bool, optional): Filter by whether country is NULL. Defaults to None.

        Returns:
            ArticlesDataset: A new dataset containing only the articles that match the
                provided criteria.
        """
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
        """
        Show distribution of query, date, event_occured, event, impact, city, state, and country
        in the filtered dataset.

        Args:
            plot (bool, optional): Whether to plot the distribution as a bar chart. Defaults to False.

        Returns:
            None
        """
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
            plt.xticks(rotation=45)
            plt.show()
            event_occured_dist.plot(kind='bar')
            plt.xticks(rotation=0)
            plt.show()
            events_dist.plot(kind='bar')
            plt.xticks(rotation=45)
            plt.show()
            impact_dist.plot(kind='bar')
            plt.xticks(rotation=0)
            plt.show()
