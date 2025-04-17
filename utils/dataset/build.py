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
        with open(failed_file_path, "a", encoding="utf-8") as f:
            while True:
                url = await failure_queue.get()
                if url is None:
                    break
                f.write(url + "\n")

    def parse_text_only(self, article: Article) -> None:
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
        success_queue: asyncio.Queue = asyncio.Queue()
        failure_queue: asyncio.Queue = asyncio.Queue()

        csv_path = os.path.join(self.links_dir, "links.csv")
        dataframe = pd.read_csv(csv_path)

        writer = asyncio.create_task(self.success_writer_task(success_queue, self.binary_file, self.index_db))
        failed_writer = asyncio.create_task(self.failed_writer_task(failure_queue, self.failed_urls_file))

        async with httpx.AsyncClient(follow_redirects=True) as session:
            processed_urls = await self.get_processed_urls()

            for start in tqdm(range(0, len(dataframe), batch_size), desc="Processing batches"):
                end = min(start + batch_size, len(dataframe))
                batch = dataframe.iloc[start:end]

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
