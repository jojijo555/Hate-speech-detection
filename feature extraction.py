from gensim.models import FastText
from pyforest import tqdm
import numpy as np
from gensim.models import FastText
# import transformers
import tqdm
import torch
from gensim.models import FastText
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, euclidean_distances
from d import create_filled_array
from d2 import similarity


class TextGraphFeatures:
    def __init__(self, text):

        if isinstance(text, str):
            self.documents = [text]
        elif isinstance(text, list) and all(isinstance(doc, str) for doc in text):
            self.documents = text
        else:
            raise ValueError("Input must be a string or list of strings")

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.graph = None

    def compute_tfidf(self):
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        return self.tfidf_matrix

    def compute_similarity_matrix(self):
        if self.tfidf_matrix is None:
            self.compute_tfidf()
        similarity_matrix1 = np.mean(cosine_similarity(self.tfidf_matrix), axis=0)
        similarity_matrix2 = np.mean(linear_kernel(self.tfidf_matrix), axis=0)
        similarity_matrix3 = np.mean(euclidean_distances(self.tfidf_matrix), axis=0)
        # Normalize and combine the matrices
        similarity_matrix = (similarity_matrix1 + similarity_matrix2 + (1 - similarity_matrix3)) / 3
        return similarity_matrix


def fast_text_embedding(texts):
    fast_T_em = []
    for i in tqdm(texts):
        tokenized_texts = [sentence.lower().split() for sentence in i]

        fasttext_model = FastText(sentences=tokenized_texts, vector_size=300, window=3, min_count=1, workers=4)

        def get_fasttext_embedding(sentence, model):
            words = sentence.lower().split()
            word_vectors = [model.wv[word] for word in words if word in model.wv]
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(model.vector_size)

        fasttext_embeddings = np.array([get_fasttext_embedding(sentence, fasttext_model) for sentence in i])
        fast_T_em.append(fasttext_embeddings)
    return fast_T_em


def Pretrained_BERT(data):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    features = []
    for text in tqdm(data, desc='Word Embedding'):
        inputs = bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[0, 0, :].detach().cpu().numpy()
        features.append(embeddings)
    return features

def hybrid_weight(text):
    if not isinstance(text, list):
        text = [text]
    simiarity = []
    for value in text:
        tf_idf = TfidfVectorizer()
        tokens = value.split()
        matrix = tf_idf.fit_transform(tokens)
        similarity_matrix1 = np.mean(cosine_similarity(matrix), axis=0)
        similarity_matrix2 = np.mean(linear_kernel(matrix), axis=0)
        similarity_matrix3 = np.mean(euclidean_distances(matrix), axis=0)
        similarity_matrix = (similarity_matrix1 + similarity_matrix2 + similarity_matrix3) / 3
        embeddin = np.array(similarity_matrix)
        # max_value = np.max(embeddin)
        # padded_array = np.full_like(embeddin, max_value)
        # pad_array = np.array(padded_array)
        simiarity.append(embeddin)
    max_len = max(len(arr) for arr in simiarity)

    # Step 2: Pad each array to max_len using column-wise max
    # First, build a 2D array with np.nan for missing entries
    matrix = np.full((len(simiarity), max_len), np.nan)
    for i, arr in enumerate(simiarity):
        matrix[i, :len(arr)] = arr

    # Step 3: Compute column-wise max, ignoring nan
    col_max = np.nanmax(matrix, axis=0)

    # Step 4: Replace nan with the column-wise max
    for i in range(len(simiarity)):
        missing = max_len - len(simiarity[i])
        if missing > 0:
            pad_values = col_max[len(simiarity[i]):]
            similarity[i] = np.concatenate([simiarity[i], pad_values])

    return simiarity

def feature_extraction(text):
    # max_value = max(simiarity, key=len)
    # converted_values = [max_value] * len(simiarity)
    # max_value = max(simiarity, key=len)
    # max_value = max(simiarity, key=len)
    # zero_array = np.zeros((3, 3))
    # for index, value in enumerate(simiarity):
    #     print(f"Index: {index}, Value: {value}")
    #     max_value = max(simiarity, key=len)
    # hybrid = hybrid_weight(text)
    embeddings = fast_text_embedding(text)
    embeddings = np.array(embeddings)
    embeddin = embeddings.reshape(-1, 1)
    P_B = Pretrained_BERT(text)
    # sim_matrix = np.array(sim_matrix).reshape(-1, 1)
    # embeddings = np.array(embeddings)
    P_B = np.array(P_B)
    # simiarit = simiarity.reshape(-1, 1)
    final_features = np.concatenate([hybrid, embeddin, P_B], axis=-1)
    return final_features


from sklearn.metrics.pairwise import cosine_similarity, linear_kernel, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


