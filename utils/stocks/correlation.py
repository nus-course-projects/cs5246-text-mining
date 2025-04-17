import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class StocksCorrelation:
    def __init__(self, incidents_df) -> None:
        self.incidents_df = incidents_df
        stocks_data_path = os.path.join("data", "stocks", "sp500_top50_5years.csv")
        self.stocks_df = pd.read_csv(stocks_data_path)
        self.stocks_df["Date"] = pd.to_datetime(self.stocks_df["Date"], format="%Y-%m-%d")
        all_tickers = self.stocks_df["Ticker"].unique()

        self.ticker_dfs = {}
        for ticker in all_tickers:
            self.ticker_dfs[ticker] = self.stocks_df[self.stocks_df["Ticker"] == ticker].reset_index(drop=True)

    def get_future_price(self, date, stock_df, days_ahead) -> float:
        future = stock_df[stock_df['Date'] >= date].head(days_ahead + 1)
        if len(future) <= days_ahead:
            return np.nan
        return future.iloc[days_ahead]['Close']

    def compute_returns(self, row, stock_df):
        base_date = row['dt']
        base_price_row = stock_df[stock_df['Date'] >= base_date].head(1)
        if base_price_row.empty:
            return pd.Series({'return_1d': np.nan, 'return_3d': np.nan, 'return_7d': np.nan})
        base_price = base_price_row.iloc[0]['Close']
        return pd.Series({
            'return_1d': (self.get_future_price(base_date, stock_df, 1) - base_price) / base_price,
            'return_3d': (self.get_future_price(base_date, stock_df, 3) - base_price) / base_price,
            'return_7d': (self.get_future_price(base_date, stock_df, 7) - base_price) / base_price,
        })
    
    def compute_correlation(self):
        all_data = []
        for ticker, stock_df in tqdm(self.ticker_dfs.items()):
            stock_df = stock_df.sort_values('Date')
            temp_incidents = self.incidents_df.copy()
            returns_df = temp_incidents.apply(lambda row: self.compute_returns(row, stock_df), axis=1)
            merged = pd.concat([temp_incidents.reset_index(drop=True), returns_df], axis=1)
            merged['ticker'] = ticker
            all_data.append(merged)

        combined_df = pd.concat(all_data)
        combined_df.dropna(subset=['return_1d', 'return_3d', 'return_7d'], inplace=True)
        event_ohe = pd.get_dummies(combined_df['event'], prefix='event')
        features_df = pd.concat([event_ohe, combined_df[['impact', 'return_1d', 'return_3d', 'return_7d']]], axis=1)

        cor_1d = features_df.corr()[['return_1d']].drop(index=['return_1d', 'return_3d', 'return_7d', 'impact'])
        cor_3d = features_df.corr()[['return_3d']].drop(index=['return_1d', 'return_3d', 'return_7d', 'impact'])
        cor_7d = features_df.corr()[['return_7d']].drop(index=['return_1d', 'return_3d', 'return_7d', 'impact'])

        return cor_1d, cor_3d, cor_7d
