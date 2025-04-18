import asyncio
import copy
import concurrent.futures
import os
from datetime import datetime, timedelta
import sqlite3
import struct
import time
from zoneinfo import ZoneInfo
from gdeltdoc import GdeltDoc, Filters  # type: ignore
import httpcore
import httpx
import pandas as pd
from newspaper import Article, ArticleException  # type: ignore
from newspaper.cleaners import DocumentCleaner  # type: ignore
from newspaper.outputformatters import OutputFormatter  # type: ignore
from tqdm import tqdm


UTC = ZoneInfo("UTC")


class DatasetBuilder:
    def __init__(self, disasters: list[str], countries: list[str]):
        """
        Initialize a DatasetBuilder.

        :param disasters: list of disaster names, e.g. ["earthquake", "flood"]
        :param countries: list of country names, e.g. ["Japan", "United States"]
        """
        self.disasters = disasters
        self.countries = countries
        self.links_dir = os.path.join("data", "news")
        os.makedirs(self.links_dir, exist_ok=True)
        self.links_raw_dir = os.path.join(self.links_dir, "raw")
        os.makedirs(self.links_raw_dir, exist_ok=True)
        self.articles_dir = os.path.join("data", "articles")
        os.makedirs(self.articles_dir, exist_ok=True)
        self.index_db = os.path.join(self.articles_dir, "index.db")
        self.retry_limit = 5
        self.failed_urls_file = os.path.join(self.articles_dir, "failed_urls.txt")
        self.binary_file = os.path.join(self.articles_dir, "news_data.bin")

    def get_news_urls_gdelt(self, query: str, start_date: datetime, end_date: datetime, country: str) -> list[dict]:
        """
        Get news URLs from GDELT Knowledge Database.

        :param query: Disaster name, e.g. "earthquake"
        :param start_date: Start date of the query
        :param end_date: End date of the query
        :param country: Country name, e.g. "Japan"
        :return: List of dictionaries containing article info
        """
        gd = GdeltDoc()

        filters = Filters(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            keyword=query,
            country=country.upper(),
            language="English"
        )
        articles = gd.article_search(filters=filters)
        all_articles = []
        for _, row in articles.iterrows():
            all_articles.append({
                "title": row["title"],
                "url": row["url"],
                "domain": row["domain"],
                "date": row["seendate"],
                "country": country,
                "query": query
            })

        return all_articles

    def fetch_news(self, year: int, month: int, country: str, disaster: str):
        """
        Fetch news articles for a given country, month, and disaster.

        Fetches news articles from GDELT Knowledge Database for a given country, month, and disaster type.
        The fetched articles are saved to a CSV file in the `links_raw_dir` directory.

        It runs at an interval of 5 seconds as that is the rate limit set by GDelt for their API.

        :param year: Year of the query
        :param month: Month of the query
        :param country: Country name, e.g. "Japan"
        :param disaster: Disaster name, e.g. "earthquake"
        :return: None
        """
        output_filename = f"{year}-{month}-{country}-{disaster}.csv"
        output_file = os.path.join(self.links_raw_dir, output_filename)

        if os.path.exists(output_file):
            print(f"Exists [{country}]-{month}/{year}-{disaster}")
            return

        _start = datetime(year=year, month=month, day=1)
        if month == 12:
            _end = datetime(year=year, month=month, day=31)
        else:
            _end = datetime(year=year, month=month + 1, day=1) - timedelta(days=1)

        today = datetime.now()
        if _end > today:
            _end = today

        print(f"Fetching [{country.upper()}] in range {_start} - {_end} for {disaster}...")
        articles = self.get_news_urls_gdelt(disaster, _start, _end, country)

        news_df = pd.DataFrame(articles)
        news_df.to_csv(output_file, index=False)

        print(f"Completed [{country}]-{month}/{year}-{disaster}")
        time.sleep(5)

    def build_news_urls_dataset(self) -> None:
        """
        Build the news URLs dataset by fetching news articles from GDelt Knowledge Database for
        all countries and disasters for each month from 2019 to 2025 (March).

        The fetched articles are saved to a CSV file in the `links_raw_dir` directory.

        After fetching all the articles, the script combines all the CSV files into one file
        named `links.csv` in the `links_dir` directory and removes duplicates.

        :return: None
        """
        def worker(task):
            self.fetch_news(*task)

        tasks = []
        for year in range(2019, 2026):
            for month in range(1, 13):
                if year == 2025 and month > 3:
                    break
                for country in self.countries:
                    for disaster in self.disasters:
                        tasks.append((year, month, country, disaster))

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(worker, tasks)

        files = os.listdir(self.links_raw_dir)
        files = [f for f in files if f.endswith(".csv")]
        files.sort()
        df_list = []
        for file in files:
            try:
                _file = os.path.join(self.links_raw_dir, file)
                _df = pd.read_csv(_file)
                df_list.append(_df)
            except pd.errors.EmptyDataError:
                continue

        df = pd.concat(df_list, ignore_index=True) if df_list else None
        if df:
            df = df.drop_duplicates(subset="url", keep="first")
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ")
            df = df.drop_duplicates(subset="url", keep="first")
            concat_links_path = os.path.join(self.links_dir, "links.csv")
            df.to_csv(concat_links_path, index=False)

            print("Done")
        else:
            print("No data found")

    def create_index_db(self) -> None:
        """
        Create the index database table if it does not exist.

        This method creates the `articles` table in the SQLite database
        specified by `self.index_db` if it does not already exist. The
        table has the following columns:

        - `id`: A unique integer primary key for each article.
        - `url`: The URL of the article.
        - `title`: The title of the article.
        - `date`: The date of the article as a Unix timestamp.
        - `country`: The country from which the article was retrieved.
        - `query`: The query used to retrieve the article.
        - `offset`: The offset of the article in the binary file.
        - `length`: The length of the article in the binary file.
        """
        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                date INTEGER,
                country TEXT,
                query TEXT,
                offset INTEGER,
                length INTEGER
            )
            """
            )
            conn.commit()

    async def success_writer_task(self, queue: asyncio.Queue, bin_file_path: str, db_path: str) -> None:
        """
        Consumer function that writes successful batches to the binary file and database.

        This method will consume the success queue and write the extracted data to the binary file
        and SQLite database. It will run until a sentinel value (None) is received in the queue, at
        which point it will exit.

        :param queue: The asyncio Queue of successful batches.
        :param bin_file_path: The path to the binary file.
        :param db_path: The path to the SQLite database.
        """
        with open(bin_file_path, "ab") as bin_file, sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            while True:
                item = await queue.get()
                if item is None:
                    break  # Shutdown signal

                url, title, date, country, query, content = item

                offset = bin_file.tell()
                encoded_content = content.encode("utf-8")
                length = len(encoded_content)
                bin_file.write(struct.pack(f"{length}s", encoded_content))

                cursor.execute(
                    "INSERT INTO articles (url, title, date, country, query, offset, length) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (url, title, date.timestamp(), country, query, offset, length),
                )
                conn.commit()

    async def failed_writer_task(self, failure_queue: asyncio.Queue, failed_file_path: str) -> None:
        """
        Consumer function that writes failed article URLs to a file.

        This method will consume the failure queue and write the URLs of the failed articles
        to a file. It will run until a sentinel value (None) is received in the queue, at
        which point it will exit.

        :param failure_queue: The asyncio Queue of failed articles.
        :param failed_file_path: The path to the file where the failed URLs will be written.
        """
        with open(failed_file_path, "a", encoding="utf-8") as f:
            while True:
                url = await failure_queue.get()
                if url is None:
                    break
                f.write(url + "\n")

    def parse_text_only(self, article: Article) -> None:
        """
        Parse an article based only on its HTML content.

        This method will parse the article based on its HTML content, and will not attempt to
        retrieve the content of any linked resources (e.g. images, stylesheets, scripts).

        :param article: The article to parse.
        """
        if not article.html:
            raise ArticleException("No HTML found")

        article.doc = article.config.get_parser().fromstring(article.html)
        article.clean_doc = copy.deepcopy(article.doc)

        if article.doc is None:
            return

        parse_candidate = article.get_parse_candidate()
        article.link_hash = parse_candidate.link_hash  # MD5

        document_cleaner = DocumentCleaner(article.config)
        output_formatter = OutputFormatter(article.config)

        article.doc = document_cleaner.clean(article.doc)

        article.top_node = article.extractor.calculate_best_node(article.doc)

        if article.top_node is not None:
            article.top_node = article.extractor.post_cleanup(article.top_node)
            article.clean_top_node = copy.deepcopy(article.top_node)

            text, article_html = output_formatter.get_formatted(article.top_node)
            article.set_article_html(article_html)  # optional
            article.set_text(text)

        article.is_parsed = True
        article.release_resources()

    async def fetch_url(self, session, url: str, retries=0):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }

        RETRY_EXCEPTIONS = (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpcore.RemoteProtocolError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx._exceptions.TransportError,
            httpx._exceptions.NetworkError,
            httpx.RequestError
        )

        try:
            response = await session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html = response.text
                if not html:
                    return None
                article = Article(url)
                article.set_html(html)
                self.parse_text_only(article)
                return article.text
            elif response.status_code in {404, 410}:
                return None
            elif response.status_code == 429 and retries < self.retry_limit:
                await asyncio.sleep(2**retries)
                return await self.fetch_url(session, url, retries + 1)
        except RETRY_EXCEPTIONS:
            if retries < self.retry_limit:
                return await self.fetch_url(session, url, retries + 1)
        return None

    async def handle_article(self, row, session, success_queue, failure_queue) -> None:
        """
        Consumer function that handles an article.

        This method will consume an article database row, fetch the article content using
        the provided session, and write either the article content to the success queue
        or the article URL to the failure queue.

        :param row: The database row for the article.
        :param session: The aiohttp ClientSession to use for fetching the article.
        :param success_queue: The asyncio Queue to write successful articles to.
        :param failure_queue: The asyncio Queue to write failed articles to.
        """
        url, title, date_str, country, query = (
            row["url"],
            row["title"],
            row["date"],
            row["country"],
            row["query"],
        )
        date = datetime.combine(
            datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date(),
            datetime(1970, 1, 1, 0, 0, 0).time(),
            tzinfo=UTC,
        )
        content = await self.fetch_url(session, url)

        if content:
            await success_queue.put((url, title, date, country, query, content))
        else:
            await failure_queue.put(url)

    async def get_processed_urls(self) -> set:
        """
        Retrieve the set of processed URLs from the database and failed URLs file.

        This asynchronous method queries the SQLite database to gather all URLs
        that have been processed and adds them to a set. It also includes any URLs
        from the failed URLs file, ensuring that all processed and failed articles
        are accounted for.

        :return: A set containing all processed and failed URLs.
        """

        processed_urls = set()

        with sqlite3.connect(self.index_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT url FROM articles")
            for row in cursor.fetchall():
                processed_urls.add(row[0])

        if os.path.exists(self.failed_urls_file):
            with open(self.failed_urls_file, "r", encoding="utf-8") as f:
                for line in f:
                    processed_urls.add(line.strip())

        return processed_urls

    async def download_articles(self, batch_size: int = 500) -> None:
        """
        Download articles from the provided list of URLs.

        This asynchronous method will download articles from the provided list of URLs,
        filter, and writing the content of each article to the database and binary file.

        :param batch_size: The number of rows to process in parallel. Defaults to 500.
        """
        success_queue: asyncio.Queue = asyncio.Queue()
        failure_queue: asyncio.Queue = asyncio.Queue()

        csv_path = os.path.join(self.links_dir, "links.csv")
        dataframe = pd.read_csv(csv_path)

        # Only extract US Published Articles
        df = dataframe[dataframe["country"] == "US"]

        # Sample while maintaining relative spread
        df['strata'] = df[['query', 'country', 'date']].astype(str).agg('-'.join, axis=1)
        query_weights = {
            'pandemic': 0.5,
            'stock market crash': 5.0
        }
        default_weight = 1.0

        group_sizes = df['strata'].value_counts()
        strata_queries = group_sizes.index.str.split('-').str[0]
        weights = strata_queries.map(lambda q: query_weights.get(q, default_weight))
        weighted_sizes = group_sizes * weights
        group_sample_sizes = (weighted_sizes / weighted_sizes.sum() * 100_000).round().astype(int)

        df['sample_size'] = df['strata'].map(group_sample_sizes)

        def sample_group(group):
            n = int(group['sample_size'].iloc[0])
            return group.sample(n=min(n, len(group)), random_state=42)

        sampled_df = df.groupby('strata', group_keys=False).apply(sample_group).reset_index(drop=True)
        sampled_df.drop(columns=['strata', 'sample_size'], inplace=True)

        writer = asyncio.create_task(self.success_writer_task(success_queue, self.binary_file, self.index_db))
        failed_writer = asyncio.create_task(self.failed_writer_task(failure_queue, self.failed_urls_file))

        async with httpx.AsyncClient(follow_redirects=True) as session:
            processed_urls = await self.get_processed_urls()

            for start in tqdm(range(0, len(sampled_df), batch_size), desc="Processing batches"):
                end = min(start + batch_size, len(sampled_df))
                batch = sampled_df.iloc[start:end]

                tasks = [
                    self.handle_article(row, session, success_queue, failure_queue)
                    for _, row in batch.iterrows()
                    if row["url"] not in processed_urls
                ]

                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Articles in batch", leave=False):
                    await coro

        await success_queue.put(None)
        await failure_queue.put(None)

        await writer
        await failed_writer

        print("Processing complete.")
