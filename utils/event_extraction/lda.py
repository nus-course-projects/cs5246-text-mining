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
from nltk.stem import WordNetLemmatizer
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
        self.lemmatizer = WordNetLemmatizer()
        self.lda_model: gensim.models.LdaModel | None = None

        # Load additional domain-specific stopwords
        self.stop_word_path = os.path.join("utils", "event_extraction", "sw1k.csv")
        self.news_stopwords = list(pd.read_csv(self.stop_word_path)['term'])

         # Merge article titles with labels (filter by available rows)
        self.df = pd.merge(articles.filtered_rows[['index_id', 'title']], articles.filtered_labels[['index_id', 'event']], on='index_id', how='inner')
        print("Pre-processing dataset...")

         # Tokenize and preprocess each title
        self.df['tokens'] = self.df['title'].progress_apply(self.preprocess)

        # Create dictionary and corpus for LDA
        self.dictionary = corpora.Dictionary(self.df['tokens'])
        self.dictionary.filter_extremes(no_below=5, no_above=0.8)

        # Convert tokens to bag-of-words format
        self.corpus = []
        for text in tqdm(self.df['tokens'], desc="Building corpus"):
            self.corpus.append(self.dictionary.doc2bow(text))
        
    def preprocess(self, text: str):
        # lowercases, removes punctuations & digits, stopwords, lemmatizes tokens
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        tokens = word_tokenize(text)
        cleaned_tokens = []
        for t in tokens:
            if t not in self.stop_words and t not in self.news_stopwords and len(t) > 2:
                t = re.sub(r'\d+', '', t)  
                if len(t) > 2: 
                    cleaned_tokens.append(self.lemmatizer.lemmatize(t))
        return cleaned_tokens


    def build_model(self, num_topics=8):
        model_dir = os.path.join("models", "event_extraction")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "lda.model5")
        if os.path.exists(model_path):
            self.lda_model = gensim.models.LdaModel.load(model_path)
            print("Loaded LDA Model")
        else:
            print("Building LDA Model...")
            self.lda_model = gensim.models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=20,
                alpha='auto'
            )
            print("Built Model")
            self.lda_model.save(model_path)

        # Assign dominant topic to each document
        def get_dominant_topic(bow):
            topics = self.lda_model.get_document_topics(bow)
            if topics:
                topics = sorted(topics, key=lambda x: -x[1]) # Sort by probability
                return topics[0][0]
            else:
                return -1
        print("Running inference...")
        self.df['extracted_event'] = [get_dominant_topic(bow) for bow in tqdm(self.corpus, desc="Running inference")]

    # Returns human-readable labels for each topic by showing top words.
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
    
    # Generate simpler topic labels using only the top 2 keywords.
    def generate_topic_labels_from_top2(self, lda_model):
        topic_labels = {}
        topics = lda_model.show_topics(num_topics=-1, num_words=5, formatted=False)
        for topic_id, topic_words in topics:
            if len(topic_words) >= 2:
                label = f"{topic_words[0][0]} + {topic_words[1][0]}"
            elif len(topic_words) == 1:
                label = topic_words[0][0]
            else:
                label = f"topic_{topic_id}"
            topic_labels[topic_id] = label
        return topic_labels
    
    # Checks if at least `min_overlap` words from the query match partially with words from the extracted topic label.
    def is_set_match_partial(self, query, extracted_event_name, min_overlap=1):
        query_words = query.lower().split()
        extracted_words = extracted_event_name.lower().replace('+', ' ').split()
        
        match_count = 0
        for qw in query_words:
            for ew in extracted_words:
                if qw in ew or ew in qw:
                    match_count += 1
                    break  
        
        return match_count >= min_overlap