import os
import requests
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.download_url = "https://github.com/nus-course-projects/cs5246-text-mining/releases/download/v0.0.1"
        os.makedirs(data_dir, exist_ok=True)
        self.block_size = 8192

    def _download_file(self, filename: str):
        url = f"{self.download_url}/{filename}"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_file = os.path.join(self.data_dir, filename)
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {filename}")
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=self.block_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

    def download(self):
        index_db = os.path.join(self.data_dir, "index.db")
        labels_db = os.path.join(self.data_dir, "labels.db")
        content_bin = os.path.join(self.data_dir, "news_data.bin")

        if os.path.exists(index_db):
            print("IndexDB already exists")
        else:
            self._download_file("index.db")

        if os.path.exists(labels_db):
            print("LabelsDB already exists")
        else:
            self._download_file("labels.db")

        if os.path.exists(content_bin):
            print("ContentBin already exists")
        else:
            self._download_file("news_data.bin")

        print("Download complete")
