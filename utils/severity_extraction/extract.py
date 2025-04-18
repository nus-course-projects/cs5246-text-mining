import os
import re
import numpy as np
from scipy.sparse import hstack, save_npz, load_npz  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
import pandas as pd
import spacy
from tqdm import tqdm
import nltk  # type: ignore
from nltk.corpus import wordnet as wn, sentiwordnet as swn  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

from utils.dataset.articles import ArticlesDataset  # type: ignore

tqdm.pandas()


class SeverityExtractor:
    def __init__(self, articles: ArticlesDataset) -> None:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('sentiwordnet')
        self.nlp = spacy.load("en_core_web_lg")

        self.preprocessed_dir = os.path.join("data", "severity_extraction")
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
                label_df_renamed[['index_id', 'impact_label']],
                on="index_id",
                how="inner"
            )
            self.articles_df['date'] = pd.to_datetime(self.articles_df['date'], unit='s').dt.date

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
        self.analyzer = SentimentIntensityAnalyzer()

    
    ### Lemmatizing the content since we are doing sentiment analysis 
    def preprocess_text(self, text):
        text = re.sub(r"\s+", " ", text)
        doc = self.nlp(text)
        tokens = [token for token in doc if not token.is_stop and not token.is_punct]
        processed = ' '.join([t.lemma_.lower() for t in tokens])
        return processed

    ### Using the VADER sentiment analysis to get the impact score
    def vader_sentiment(self) -> tuple:
        print("Running Sentiment Analysis")

        def get_impact_score(text):
            sentiment = self.analyzer.polarity_scores(text)
            compound = sentiment["compound"]

            if compound <= -0.7:
                return 5
            elif compound <= -0.4:
                return 4
            elif compound <= -0.2:
                return 3
            elif compound <= 0.3:
                return 2
            elif compound < 0.3:
                return 1
            else:
                return 0

        preds = self.articles_df["content_preprocessed"].progress_apply(get_impact_score)
        gt = self.articles_df["impact_label"]
        return gt, preds

    def _build_for_ml(self) -> tuple:
        x_train_file = os.path.join(self.preprocessed_dir, "x_train.npz")
        y_train_file = os.path.join(self.preprocessed_dir, "y_train.npy")
        x_test_file = os.path.join(self.preprocessed_dir, "x_test.npz")
        y_test_file = os.path.join(self.preprocessed_dir, "y_test.npy")
        if os.path.exists(x_train_file):
            x_train_vectorized = load_npz(x_train_file)
            y_train = np.load(y_train_file)
            x_test_vectorized = load_npz(x_test_file)
            y_test = np.load(y_test_file)
            print("Loaded test and train vectors")
        else:
            ### Splitting the data into train and test sets
            x_train, x_test = train_test_split(self.articles_df, test_size=0.3, train_size=0.7, random_state=100, shuffle=True, stratify=self.articles_df["impact_label"])
            tfidf_vectorizer = TfidfVectorizer(max_features=10000)  ### Convert the text to a matrix of TF-IDF features
            x_train_vectorized = tfidf_vectorizer.fit_transform(x_train["content_preprocessed"])
            x_test_vectorized = tfidf_vectorizer.transform(x_test["content_preprocessed"])
            y_train = x_train["impact_label"]
            y_test = x_test["impact_label"]

            x_train["content_length"] = x_train["content"].apply(lambda x: len(x))
            x_test["content_length"] = x_test["content"].apply(lambda x: len(x))


            ### Spacy's and wordnet's pos tags are different so we create a mapping
            def spacy_to_wordnet_pos(spacy_pos: str):
                return {
                    "ADJ": wn.ADJ,
                    "VERB": wn.VERB,
                    "NOUN": wn.NOUN,
                    "ADV": wn.ADV
                }.get(spacy_pos, None)


            ### Count the number of positive and negative words in the content using
            ### SentiWordNet
            def count_sentiment(articles_df):
                articles_df["pos_num"] = 0
                articles_df["neg_num"] = 0

                for idx, row in tqdm(articles_df.iterrows()):
                    pos = 0
                    neg = 0
                    text = row["content_preprocessed"]
                    doc = self.nlp(text)
                    for token in doc:
                        wn_pos = spacy_to_wordnet_pos(token.pos_)
                        if not wn_pos:
                            continue
                        synsets = wn.synsets(token.lemma_, pos=wn_pos)
                        if not synsets:
                            continue
                        swn_syn = swn.senti_synset(synsets[0].name())
                        if swn_syn.pos_score() > swn_syn.neg_score():
                            pos += 1
                        elif swn_syn.neg_score() > swn_syn.pos_score():
                            neg += 1
                    articles_df.at[idx, "pos_wrds"] = pos
                    articles_df.at[idx, "neg_wrds"] = neg
                return articles_df

            x_train = count_sentiment(x_train)
            x_test = count_sentiment(x_test)

            content_length = x_train["content_length"].values.reshape(-1, 1)
            pos_wrds = x_train["pos_wrds"].values.reshape(-1, 1)
            neg_wrds = x_train["neg_wrds"].values.reshape(-1, 1)
            x_train_vectorized = hstack([x_train_vectorized, content_length, pos_wrds, neg_wrds])  ### added handcrafted features

            content_length = x_test["content_length"].values.reshape(-1, 1)
            pos_wrds = x_test["pos_wrds"].values.reshape(-1, 1)
            neg_wrds = x_test["neg_wrds"].values.reshape(-1, 1)
            x_test_vectorized = hstack([x_test_vectorized, content_length, pos_wrds, neg_wrds])   ### added handcrafted features

            save_npz(x_train_file, x_train_vectorized)
            save_npz(x_test_file, x_test_vectorized)
            np.save(y_train_file, y_train)
            np.save(y_test_file, y_test)
            print("Built test and train vectors")

        return x_train_vectorized, y_train, x_test_vectorized, y_test


    ### Train and predict using Naive Bayes
    def naive_bayes(self) -> tuple:
        print("Training Naive Bayes Classifier")
        x_train, y_train, x_test, y_test = self._build_for_ml()
        nb = MultinomialNB()
        nb.fit(x_train, y_train)
        y_pred = nb.predict(x_test)
        return y_test, y_pred

    ### Train and predict using Logistic Regression
    def logistic_regression(self) -> tuple:
        print("Training Logistic Regression Classifier")
        x_train, y_train, x_test, y_test = self._build_for_ml()
        log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
        log_reg.fit(x_train, list(y_train))
        y_pred = log_reg.predict(x_test)
        y_pred = list(y_pred)
        return y_test, y_pred

    ### Train and predict using Random Forest
    def random_forest(self) -> tuple:
        print("Training Random Forest Classifier")
        x_train, y_train, x_test, y_test = self._build_for_ml()
        rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=12)
        rf_model.fit(x_train, list(y_train))
        y_pred = rf_model.predict(x_test)
        return y_test, y_pred
