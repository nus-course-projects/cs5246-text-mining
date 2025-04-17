import os
import re
import pandas as pd
import gensim  # type: ignore
from gensim import corpora  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
import nltk  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore
from tqdm import tqdm

from utils.dataset.articles import ArticlesDataset
tqdm.pandas()


class LDA:
    def __init__(self, articles: ArticlesDataset) -> None:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lda_model: gensim.models.LdaModel | None = None
        self.df = pd.merge(articles.filtered_rows[['index_id', 'title']], articles.filtered_labels[['index_id', 'event']], on='index_id', how='inner')
        print("Pre-processing dataset...")
        self.df['tokens'] = self.df['title'].progress_apply(self.preprocess)
        self.dictionary = corpora.Dictionary(self.df['tokens'])
        self.dictionary.filter_extremes(no_below=5, no_above=0.8)
        self.corpus = []
        for text in tqdm(self.df['tokens'], desc="Building corpus"):
            self.corpus.append(self.dictionary.doc2bow(text))

    def preprocess(self, text: str):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return tokens

    def build_model(self, num_topics=10):
        model_dir = os.path.join("models", "event_extraction")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "lda.model")
        if os.path.exists(model_path):
            self.lda_model = gensim.models.LdaModel.load(model_path)
            print("Loaded LDA Model")
        else:
            print("Building LDA Model...")
            self.lda_model = gensim.models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=95,
                passes=20,
                alpha='auto'
            )
            print("Built Model")
            self.lda_model.save(model_path)

        def get_dominant_topic(bow):
            topics = self.lda_model.get_document_topics(bow)
            if topics:
                topics = sorted(topics, key=lambda x: -x[1])
                return topics[0][0]
            else:
                return -1
        print("Running inference...")
        self.df['extracted_event'] = [get_dominant_topic(bow) for bow in tqdm(self.corpus, desc="Running inference")]

    def generate_topic_labels(self):
        if not self.lda_model:
            raise RuntimeError("LDA model not built yet")
        print("Generating topics...")
        topic_labels = {}
        topics = self.lda_model.show_topics(num_topics=-1, num_words=5, formatted=False)
        for topic_id, topic_words in topics:
            label = ", ".join([f"{word} ({weight:.2f})" for word, weight in topic_words])
            topic_labels[topic_id] = label
        return topic_labels
