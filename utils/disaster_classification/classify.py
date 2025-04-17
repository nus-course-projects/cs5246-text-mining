import os
from sentence_transformers import SentenceTransformer
import contractions  # type: ignore
import spacy
import numpy as np
import regex as re
import joblib  # type: ignore
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, log_loss  # type: ignore
from utils.dataset.articles import ArticlesDataset


class TextPipeline:
    def __init__(self, embeddings: SentenceTransformer, lemmatize: bool = True):
        self.embeddings = embeddings
        self.lemmatize = lemmatize
        if lemmatize:
            # self.nlp = spacy.load("en_core_web_md")
            self.nlp = spacy.load("en_core_web_lg")

    def preprocess(self, text: str) -> str:
        _text = text.strip()
        _text = contractions.fix(_text)
        _text = _text.lower().strip()
        # _text = _text.translate(str.maketrans("", "", string.punctuation))
        _text = re.sub(r'[\p{P}+]', '', _text)
        _text = re.sub(r"\s+", " ", _text)

        if self.lemmatize:
            doc = self.nlp(_text)
            words = [token.lemma_ for token in doc if not token.is_stop]
            combined = " ".join(words)
            return combined
        return _text

    def embed(self, text: str):
        return self.embeddings.encode(text)

    def run(self, text: str, show: bool = False):
        content = self.preprocess(text)
        if show:
            print(content)
        return self.embed(content)


class Dataset:
    def __init__(self, articles: ArticlesDataset, text_pipeline: TextPipeline):
        self.articles = articles
        self.text_pipeline = text_pipeline

        data_dir = os.path.join("data", "disaster_classification")
        os.makedirs(data_dir, exist_ok=True)
        x_file = os.path.join(data_dir, "x.npy")
        y_file = os.path.join(data_dir, "y.npy")
        if os.path.exists(x_file) and os.path.exists(y_file):
            self.X = np.load(x_file)
            print(f"Loaded X: {self.X.shape}")
            self.Y = np.load(y_file)
            print(f"Loaded Y: {self.Y.shape}")
            return

        _x_data = []
        _y_data = []
        print("Pre-processing articles")
        for metadata, content, label in tqdm(articles):  # type: ignore
            _x_data.append(text_pipeline.run(metadata["title"] + ' ' + content))
            _y_data.append(label["event_occured"])

        self.X = np.array(_x_data)
        self.Y = np.array(_y_data).astype(int)

        os.makedirs(data_dir, exist_ok=True)
        np.save(x_file, self.X)
        print(f"Saved X: {self.X.shape}")
        np.save(y_file, self.Y)
        print(f"Saved Y: {self.Y.shape}")

    def get_split(self, test_size: float = 0.2, random_state: int = 103):
        return train_test_split(self.X, self.Y, test_size=test_size, random_state=random_state, stratify=self.Y)


class Model:
    def __init__(self, dataset: Dataset, k: int = 5, random_state: int = 103):
        self.dataset = dataset
        self.model = LogisticRegression(max_iter=1000)
        self.kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        self.model_dir = os.path.join("models", "disaster_classification")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_file = os.path.join(self.model_dir, "logistic.joblib")

    def load(self) -> None:
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            print("Loaded model")
            return
        raise FileNotFoundError("Model not found")

    def save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_file)
        print("Saved model")

    def train(self, split_random_state: int = 103):
        X_train, X_test, y_train, y_test = self.dataset.get_split(test_size=0.2, random_state=split_random_state)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for fold, (train_index, val_index) in enumerate(tqdm(self.kf.split(X_train), total=self.kf.get_n_splits(), desc="Training"), 1):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            Y_train_fold, Y_val_fold = y_train[train_index], y_train[val_index]
            self.model.fit(X_train_fold, Y_train_fold)
            Y_train_proba = self.model.predict_proba(X_train_fold)
            Y_val_proba = self.model.predict_proba(X_val_fold)
            Y_train_pred = self.model.predict(X_train_fold)
            Y_val_pred = self.model.predict(X_val_fold)

            train_loss = log_loss(Y_train_fold, Y_train_proba)
            val_loss = log_loss(Y_val_fold, Y_val_proba)
            train_acc = accuracy_score(Y_train_fold, Y_train_pred)
            val_acc = accuracy_score(Y_val_fold, Y_val_pred)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            tqdm.write(f"Fold {fold}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        print(f"Mean Train Loss={np.mean(train_losses):.4f}, Mean Train Acc={np.mean(train_accuracies):.4f}, Mean Val Loss={np.mean(val_losses):.4f}, Mean Val Acc={np.mean(val_accuracies):.4f}")
        print("Finished Model Training")
        self.save()

        print("Testing Model")
        Y_test_proba = self.model.predict_proba(X_test)
        Y_test_pred = self.model.predict(X_test)
        test_loss = log_loss(y_test, Y_test_proba)
        test_acc = accuracy_score(y_test, Y_test_pred)
        print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        return train_losses, train_accuracies, val_losses, val_accuracies


class DisasterClassifier:
    def __init__(self, articles: ArticlesDataset, k_fold: int = 5, random_state: int = 103):
        self.articles = articles
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_pipeline = TextPipeline(self.embeddings)
        self.dataset = Dataset(self.articles, self.text_pipeline)
        self.model = Model(self.dataset, k=k_fold, random_state=random_state)

    def train(self, split_random_state: int = 103):
        return self.model.train(split_random_state)
