"""module for keeping logic for topic modelling"""
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd


def fit(path_to_file: str, name_text_column: str):
    """method which provide document-term matrix"""
    token_number: dict = {}
    token_quantity: dict = {}
    counters_docs = []
    for line in pd.read_csv(path_to_file, chunksize=1, sep=","):
        line = clean_text(line[name_text_column])
        token_number, token_quantity = get_token_dictionaries(
            line, token_number, token_quantity
        )
        counters_docs.append(Counter(line))
    document_term_mtx = get_document_term_matrix(counters_docs, token_number)
    matrix, _, _ = randomized_svd(
        document_term_mtx, n_components=5, n_iter=5, random_state=None
    )
    return matrix


def predict(matrix, index: int, number_neighbors: int):
    """method which will return indexes the closest documents"""
    return get_neighbors(index, matrix, number_neighbors)


def get_base_for_matrix(counter_docs, token_number):
    """nested method"""
    ar_row = []
    ar_col = []
    ar_val = []
    for idx, item in enumerate(counter_docs):
        ar_row.extend(np.repeat(idx, len(item)))
        ar_col.extend([token_number.get(token) for token in item.keys()])
        ar_val.extend(list(item.values()))
    return ar_row, ar_col, ar_val


def get_token_dictionaries(line, token_number, token_quantity):
    """nested method"""
    counter = 0
    for word_item in line:
        if word_item not in token_number.keys():
            token_number[word_item] = counter
            counter += 1
        token_quantity[word_item] = token_quantity.get(word_item, 0) + 1
    return token_number, token_quantity


def get_neighbors(idx, matrix, number_of_neighbors):
    """nested method"""
    norms = LA.norm(matrix, axis=1, keepdims=True)
    matrix /= norms
    doc = matrix[idx]
    mult = np.matmul(matrix, doc)
    return np.argpartition(a=mult, kth=-number_of_neighbors)[-number_of_neighbors:]


def get_document_term_matrix(counter_docs, token_number):
    """nested method"""
    ar_row, ar_col, ar_val = get_base_for_matrix(counter_docs, token_number)
    sparse_mtx = csr_matrix(
        (ar_val, (ar_row, ar_col)), shape=(len(counter_docs), len(token_number))
    )
    return get_tf_idf_matrix(sparse_mtx)


def get_tokens_from_doc(doc):
    """nested method"""
    return set(doc)


def count_quantity_token_in_doc(doc, token):
    """nested method"""
    counter = 0
    for item in doc:
        if item == token:
            counter += 1
    return counter


def get_tf_idf_matrix(matrix):
    """nested method"""
    number_of_words_in_doc = np.matrix(matrix.getnnz(axis=1)).transpose()
    tf_part = matrix.multiply(1 / number_of_words_in_doc)
    number_docs = matrix.shape[0]
    item = matrix.getnnz(axis=0)
    dataframe = tf_part.multiply(np.log(number_docs / item))
    return tf_part.multiply(dataframe)


def clean_text(corpus: pd.Series):
    """nested method"""
    porter_stemmer = PorterStemmer()
    corpus = corpus.apply(remove_punctuation)
    corpus = corpus.apply(tokenization)
    corpus = corpus.apply(remove_stopwords)
    corpus = corpus.apply(lambda x: [item.lower() for item in x])
    corpus = corpus.apply(lambda x: [porter_stemmer.stem(word) for word in x])
    return corpus.to_numpy()[0]


def remove_punctuation(text):
    """nested method"""
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free


def tokenization(text):
    """nested method"""
    tokens = re.split(r"\W+", text)
    return tokens


def remove_stopwords(text):
    """nested method"""
    output = [i for i in text if i not in stopwords.words("english")]
    return output


def get_correctness_doc_count(
    dataframe, index_doc, matrix, number_of_neighbors, name_of_target
):
    """method wich will return value of metric"""
    neighbors = get_neighbors(index_doc, matrix, number_of_neighbors)
    counter = 0
    for item in neighbors:
        if dataframe[name_of_target][item] == dataframe[name_of_target][index_doc]:
            counter += 1
    return counter / number_of_neighbors
