import json
import os
import re
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import tiktoken

from utils.dataset.articles import ArticlesDataset


TEMPLATE = """
You are an expert at extracting structured information about events from news articles.

Given the titles, dates, and content of multiple news articles, you must determine if a major event occurred in each one. If an event occurred, you should extract:
- The **type of event** (earthquake, flood, etc.)
- The **exact date** of the event in ISO 8601 format (e.g., "2023-11-04T09:15:00Z")
- The **location** of the event (e.g., "JAPAN")

You should return a **single JSON array** with the results for all articles. Each item in the array should contain:
  - "event_occured" (boolean): whether an event occurred or not.
  - "event" (string): type of event (if any).
  - "impact" (integer): a number 1-5 denoting the impact (if any information present). 5 being high impact, 1 being low impact. If there is no information present for impact, use 0.
  - "dt" (string): date of the event (ISO 8601 format).
  - "loc" (string): location of the event.

If no event occurred in an article, it's corresponding object in the JSON array should be:
{{
  "event_occured": false
}}

You should return a **single JSON array** with the results for all articles.

The events to look for include:
{events}

Now analyze the following articles:

{articles}
""".strip()


class LLM:
    def __init__(self, model_name: str):
        load_dotenv(".env")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=0
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        prompt_template = PromptTemplate(
            input_variables=["articles"],
            template=TEMPLATE
        )
        event_list = [
            "earthquake", "flood", "hurricane", "wildfire", "tsunami", "landslide", "volcano", "tornado", "drought",
            "explosion", "fires", "terrorist attack", "plane crash", "train derailment", "oil spill", "building collapse",
            "nuclear accident", "pandemic", "cyber attack", "stock market crash", "power outage", "riots", "protest"
        ]

        self.formatted_prompt = prompt_template.partial(
            events=", ".join(event_list),
        )

        self.history: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def count_tokens(self, text: str):
        return len(self.encoding.encode(text))

    def analyze_article(self, articles):
        try:
            articles_str = ""
            for article in articles:
                articles_str += f"Title: {article['title']}\nPublish Date: {article['date'].strftime('%Y-%m-%d')}\nContent:\n{article['content']}\n\n"

            prompt = self.formatted_prompt.format(articles=articles_str)
            estimate_input_tokens = self.count_tokens(prompt)
            result = self.llm.invoke(prompt)
            content = result.content
            usage = result.usage_metadata
            estimate_output_tokens = self.count_tokens(content)
            cleaned = re.sub(r"^```(?:json)?\s*|```$", "", content.strip(), flags=re.IGNORECASE | re.MULTILINE)

            true_input_tokens = usage["input_tokens"]
            true_output_tokens = usage["output_tokens"]
            self.history.append({
                "estimate_input_tokens": estimate_input_tokens,
                "estimate_output_tokens": estimate_output_tokens,
                "true_input_tokens": true_input_tokens,
                "true_output_tokens": true_output_tokens
            })
            _pobj = json.loads(cleaned)
            if len(_pobj) != len(articles):
                print(f"Error parsing: Got {len(_pobj)} responses != {len(articles)} articles")
                return None
            return _pobj
        except Exception as e:
            print("Error parsing:", e)
            print("Raw Output: ", result)
            return None


class LabelsBuilder:
    def __init__(self) -> None:
        self.data_dir = os.path.join("data", "articles")
        self.dataset = ArticlesDataset(self.data_dir)
        self.index_db = "index.db"
        self.bin_file = "news_data.bin"
        self.batch_size = 10
        self.num_workers = 5
        self.output_file = "labels.json"
        self.failed_batches_file = "falied_batches.json"
        self.success_queue = queue.Queue()  # type: ignore
        self.failure_queue = queue.Queue()  # type: ignore

        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding="utf-8") as fr:
                self.extracted_data = json.load(fr)
        else:
            self.extracted_data = []

        if os.path.exists(self.failed_batches_file):
            with open(self.failed_batches_file, 'r', encoding="utf-8") as fr:
                self.failed_batches = json.load(fr)
        else:
            self.failed_batches = []

        self.processed_indices = set([item["index"] for item in self.extracted_data])
        self.failed_batch_indices = set(self.failed_batches)

    def success_writer(self):
        while True:
            batch = self.success_queue.get()
            if batch is None:
                break
            print("[SUCCESS] Writing batch...")

            self.extracted_data.extend(batch)
            with open(self.output_file, 'w', encoding="utf-8") as fw:
                json.dump(self.extracted_data, fw, indent=2)
            self.success_queue.task_done()

    def failure_writer(self):
        while True:
            batch_index = self.failure_queue.get()
            if batch_index is None:
                break
            print("[FAILED] Writing batch...")

            self.failed_batches.append(batch_index)
            with open(self.failed_batches_file, 'w', encoding="utf-8") as fw:
                json.dump(self.failed_batches, fw, indent=2)
            self.failure_queue.task_done()

    def process_batch(self, _batch_start):
        llm = LLM("gemini-2.5-pro-preview-03-25")

        batch_end = min(_batch_start + self.batch_size, len(self.dataset))
        batch_index = _batch_start // self.batch_size

        # if batch_index in failed_batch_indices:
        #     return

        if all(i in self.processed_indices for i in range(_batch_start, batch_end)):
            print(f"[Batch {batch_index}] All indices processed. Skipping...")
            return

        articles = []
        indices = []

        for i in range(_batch_start, batch_end):
            if i in self.processed_indices:
                continue
            metadata, content = self.dataset[i]
            articles.append({
                "title": metadata["title"],
                "date": metadata["date"],
                "content": content
            })
            indices.append(i)

        if not articles:
            return

        try:
            extracted = llm.analyze_article(articles)
            if extracted is None:
                raise ValueError("LLM returned None")

            for idx, result in zip(indices, extracted):
                result["index"] = idx
            print(f"[Batch {batch_index}] Success")
            self.success_queue.put(extracted)

        except Exception as e:
            print(f"[Batch {batch_index}] Failed: {e}")
            self.failure_queue.put(batch_index)

    def build_labels(self) -> None:
        success_writer_thread = threading.Thread(target=self.success_writer, daemon=True)
        failure_writer_thread = threading.Thread(target=self.failure_writer, daemon=True)
        success_writer_thread.start()
        failure_writer_thread.start()

        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            try:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(self.process_batch, batch_start) for batch_start in range(0, len(self.dataset), self.batch_size)]
                    for future in futures:
                        future.result()
                        pbar.update(1)

            except KeyboardInterrupt:
                print("\nInterrupted by user. Shutting down gracefully...")
                self.success_queue.join()
                self.failure_queue.join()
                print("Queues joined")

                self.success_queue.put(None)
                self.failure_queue.put(None)
                print("Waiting for writer threads to finish...")

                success_writer_thread.join()
                failure_writer_thread.join()
                print("Threads joined")

        print("Done! Joining Writers...")
        self.success_queue.join()
        self.failure_queue.join()

        self.success_queue.put(None)
        self.failure_queue.put(None)

        print("Waiting for writer threads to finish...")
        success_writer_thread.join()
        failure_writer_thread.join()
