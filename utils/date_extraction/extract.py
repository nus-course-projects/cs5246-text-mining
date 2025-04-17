from datetime import datetime
import re
import os
import dateparser
import pandas as pd
import spacy
from tqdm import tqdm

from utils.dataset.articles import ArticlesDataset


tqdm.pandas()


class DateExtractor:
    def __init__(self, articles: ArticlesDataset) -> None:
        # # self.nlp = spacy.load("en_core_web_md")
        self.nlp = spacy.load("en_core_web_lg")

        self.preprocessed_dir = os.path.join("data", "date_extraction")
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        preprocessed_csv = os.path.join(self.preprocessed_dir, "articles_preprocessed.csv")
        if os.path.exists(preprocessed_csv):
            self.articles_df = pd.read_csv(preprocessed_csv)
        else:
            print("Building dataframe...")
            label_df_renamed = articles.filtered_labels.rename(
                columns={col: f"{col}_label" for col in articles.filtered_labels.columns if col != "index_id"}
            )
            self.articles_df = pd.merge(
                articles.filtered_rows[['index_id', 'title', 'date', 'country', 'query']],
                label_df_renamed[['index_id', 'dt_label', 'impact_label']],
                on="index_id",
                how="inner"
            )
            self.articles_df['date'] = pd.to_datetime(self.articles_df['date'], unit='s').dt.date
            self.articles_df["dt_label"] = pd.to_datetime(self.articles_df["dt_label"], errors="coerce")
            self.articles_df["dt_label"] = self.articles_df["dt_label"].dt.date

            contents = []
            contents_preprocessed = []
            for idx in tqdm(range(len(articles))):
                _, content, _ = articles[idx]
                contents.append(content)
                contents_preprocessed.append(self.preprocess_text(content))

            self.articles_df['content'] = contents
            self.articles_df['content_preprocessed'] = contents_preprocessed

            self.articles_df.to_csv(preprocessed_csv, index=False)

        print(f"Loaded {self.articles_df.shape[0]} articles")

        self.articles_df["dates_content"] = [[] for _ in range(len(self.articles_df))]
        self.articles_df["dates_title"] = [[] for _ in range(len(self.articles_df))]

    def preprocess_text(self, text):
        text = re.sub(r"\s+", " ", text)
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        processed = ' '.join([t.text for t in tokens])
        return processed

    def extract_date_entities(self) -> None:
        print("Building candidate date entities")
        extracted_csv = os.path.join(self.preprocessed_dir, "candidated.csv")
        if os.path.exists(extracted_csv):
            self.articles_df = pd.read_csv(extracted_csv)
            print("Loaded saved candidates")
        else:
            for index, row in tqdm(self.articles_df.iterrows(), total=len(self.articles_df)):
                dates_content = []
                dates_title = []
                doc = self.nlp(row["content_preprocessed"])
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        dates_content.append(ent.text)

                doc = self.nlp(row["title"])
                for ent in doc.ents:
                    if ent.label_ == "DATE":
                        dates_title.append(ent.text)

                self.articles_df.at[index, "dates_content"] = dates_content
                self.articles_df.at[index, "dates_title"] = dates_title
            self.articles_df.to_csv(extracted_csv, index=False)

    def _extract_dates(self, dates_title, dates_content, publication_date) -> None:
        days_of_the_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        _pub = datetime.strptime(publication_date, "%Y-%m-%d")
        publication_date = datetime.combine(_pub, datetime.min.time())

        def parse_all(dates):
            parsed_dates = []
            for d in dates:
                parsed = dateparser.parse(d, settings={"RELATIVE_BASE": publication_date, "PREFER_DATES_FROM": "past"})
                if parsed is None:
                    continue
                parsed = parsed.replace(tzinfo=None)
                if d.lower().strip() in days_of_the_week:
                    if parsed.weekday() == publication_date.weekday():
                        parsed = publication_date
                parsed_dates.append(parsed)
            return parsed_dates

        parsed_content = parse_all(dates_content)
        parsed_title = parse_all(dates_title)

        if parsed_content:
            for dt in parsed_content:
                if dt and dt <= publication_date:
                    return dt

        if parsed_title:
            for dt in parsed_title:
                if dt and dt <= publication_date:
                    return dt
        return None

    def extract_dates(self):
        extracted_csv = os.path.join(self.preprocessed_dir, "extracted.csv")
        if os.path.exists(extracted_csv):
            self.articles_df = pd.read_csv(extracted_csv)
            print("Loaded saved extracted dates")
        else:
            print("Extracting event dates")
            self.articles_df["extracted_date"] = self.articles_df.progress_apply(lambda row: self._extract_dates(row["dates_title"], row["dates_content"], publication_date=row["date"]), axis=1)  # type: ignore
            self.articles_df["extracted_date"] = pd.to_datetime(self.articles_df["extracted_date"], errors="coerce")
            self.articles_df["extracted_date"] = self.articles_df["extracted_date"].dt.date
            self.articles_df.to_csv(extracted_csv, index=False)

        accuracy = (self.articles_df["extracted_date"] == self.articles_df["dt_label"]).mean()
        return accuracy
