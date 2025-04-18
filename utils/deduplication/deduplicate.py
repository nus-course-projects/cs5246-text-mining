import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore
from fuzzywuzzy import fuzz  # type: ignore


class Deduplicator:

    def __init__(self, labels_df: pd.DataFrame):
        self.df = labels_df
        # Drop rows where the latitude, longitude column is empty
        self.df = self.df.dropna(subset=['latitude', 'longitude', 'dt'])
        self.df["dt"] = pd.to_datetime(self.df["dt"], format="%Y-%m-%d", errors="coerce")
        self.unique_events_df: pd.DataFrame | None = None

    def deduplicate(self, spatial_radius: int = 100, temporal_radius: int = 3):
        """
        Deduplicates the labels dataframe by spatially and temporally clustering similar
        reports. The clustering is done with DBSCAN for spatial clustering and a custom
        algorithm for temporal clustering. The algorithm assigns a unique event_id to each
        group of reports. The confidence score for each group is calculated by taking into
        account the number of reports, the standard deviation of the impact, latitude, longitude,
        and date span of the reports.

        :param spatial_radius: The radius (in kilometers) to use for spatial clustering.
        :param temporal_radius: The radius (in days) to use for temporal clustering.
        :return: A dataframe with the unique events and their corresponding confidence scores.
        """

        # Spatial Clustering
        coords = np.radians(self.df[['latitude', 'longitude']].values)
        spatial_cluster = DBSCAN(eps=spatial_radius / 6371, min_samples=1, metric='haversine')
        self.df['loc_cluster'] = spatial_cluster.fit_predict(coords)

        # Temporal Clustering
        self.df = self.df.sort_values('dt')
        self.df['time_cluster'] = -1

        for (_event, _loc_cluster), group in self.df.groupby(['event', 'loc_cluster']):
            dates = group['dt'].tolist()
            cluster_ids = []
            cluster_id = 0
            last_date = dates[0]

            for date in dates:
                if (date - last_date).days > temporal_radius:
                    cluster_id += 1
                cluster_ids.append(cluster_id)
                last_date = date

            self.df.loc[group.index, 'time_cluster'] = cluster_ids

        self.df['event_id'] = self.df.apply(lambda row: f"{row['event']}_{row['loc_cluster']}_{row['time_cluster']}", axis=1)
        unique_events = []
        for event_id, group in self.df.groupby('event_id'):
            # Confidence score calculation
            n_reports = len(group)
            impact_std = group['impact'].std(ddof=0) if len(group['impact']) > 1 else 0
            lat_std = group['latitude'].std(ddof=0)
            lon_std = group['longitude'].std(ddof=0)
            date_span = (group['dt'].max() - group['dt'].min()).days

            loc_consistency = np.mean([
                fuzz.partial_ratio(a, b)
                for i, a in enumerate(group['loc'])
                for j, b in enumerate(group['loc']) if i < j
            ]) if len(group) > 1 else 100

            # Normalize components for score (higher is better)
            score = (
                min(n_reports / 10, 1.0) * 0.4 +
                max(1 - impact_std / 2, 0) * 0.2 +
                max(1 - lat_std / 0.1, 0) * 0.1 +
                max(1 - lon_std / 0.1, 0) * 0.1 +
                max(1 - date_span / 5, 0) * 0.1 +
                (loc_consistency / 100) * 0.1
            )
            score = round(min(score, 1.0), 3)

            unique_event = {
                'event_id': event_id,
                'event': group['event'].iloc[0],
                'dt': group['dt'].min(),
                'impact': group['impact'].max(),
                'latitude': group['latitude'].mean(),
                'longitude': group['longitude'].mean(),
                'city': group['city'].mode().iloc[0] if not group['city'].mode().empty else None,
                'state': group['state'].mode().iloc[0] if not group['state'].mode().empty else None,
                'country': group['country'].mode().iloc[0] if not group['country'].mode().empty else None,
                'confidence': score,
                'n_reports': n_reports
            }
            unique_events.append(unique_event)

        self.unique_events_df = pd.DataFrame(unique_events)

    def get_unique_events(self, confidence_threshold: float = 0.5):
        if self.unique_events_df is None:
            raise ValueError("Deduplication must be performed first.")
        return self.unique_events_df[self.unique_events_df['confidence'] >= confidence_threshold]
