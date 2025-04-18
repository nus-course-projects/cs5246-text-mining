import os
import requests
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, data_dir: str, stocks_dir: str):
        """
        Initialize the dataset downloader.

        Args:
            data_dir (str): The directory to store the dataset files.
            stocks_dir (str): The directory to store the stock data files.

        Attributes:
            data_dir (str): The directory to store the dataset files.
            stocks_dir (str): The directory to store the stock data files.
            download_url (str): The URL for downloading dataset files.
            block_size (int): The block size for downloading files.
        """
        self.data_dir = data_dir
        self.stocks_dir = stocks_dir
        self.download_url = "https://github.com/nus-course-projects/cs5246-text-mining/releases/download/v0.0.1"
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(stocks_dir, exist_ok=True)
        self.block_size = 8192

    def _download_file(self, filename: str, download_dir: str):
        """
        Downloads a file from the specified URL and saves it to the given directory.

        Args:
            filename (str): The name of the file to download.
            download_dir (str): The directory where the file will be saved.

        Raises:
            requests.exceptions.RequestException: If the download fails due to a network error.
        """
        url = f"{self.download_url}/{filename}"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_file = os.path.join(download_dir, filename)
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {filename}")
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=self.block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

    def download(self):
        """
        Download the dataset files and stock data files if they do not exist.

        Downloads the following files if they do not exist:
            - IndexDB (index.db)
            - LabelsDB (labels.db)
            - ContentBin (news_data.bin)
            - Stocks CSV (sp500_top50_5years.csv)

        If a file already exists, it will not be downloaded.
        """
        index_db = os.path.join(self.data_dir, "index.db")
        labels_db = os.path.join(self.data_dir, "labels.db")
        content_bin = os.path.join(self.data_dir, "news_data.bin")
        stocks_csv = os.path.join(self.stocks_dir, "sp500_top50_5years.csv")

        if os.path.exists(index_db):
            print("IndexDB already exists")
        else:
            self._download_file("index.db", self.data_dir)

        if os.path.exists(labels_db):
            print("LabelsDB already exists")
        else:
            self._download_file("labels.db", self.data_dir)

        if os.path.exists(content_bin):
            print("ContentBin already exists")
        else:
            self._download_file("news_data.bin", self.data_dir)

        if os.path.exists(stocks_csv):
            print("Stocks CSV Already Exists")
        else:
            self._download_file("sp500_top50_5years.csv", self.stocks_dir)

        print("Download complete")
