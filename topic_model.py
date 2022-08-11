"""module for keeping logic for topic modelling"""
import re
import string
from collections import Counter
from xmlrpc.client import Boolean

import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd


def fit(path_to_file: str, number_text_column: int, n_components: int, n_iter: int, random_state, combined_value: Boolean, quantity_threshold_up:int, quantity_threshold_down:int):
    """method which provide document-term matrix"""
    token_number: dict = {}
    token_counter = Counter()
    counters_docs = []
    with open(path_to_file, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for line in reader:
            line = clean_text(line[number_text_column])
            for word_item in line:
                token_number.setdefault(word_item, len(token_number))
            counter = Counter(line)
            token_counter += counter    
            counters_docs.append(counter)
    token_quantity = dict(token_counter)
    document_term_mtx = get_document_term_matrix(counters_docs, token_number, quantity_threshold_up, quantity_threshold_down, token_quantity)
    matrix_U, matrix_S, matrix_V = randomized_svd(
        document_term_mtx, n_components=n_components, n_iter=n_iter, random_state=random_state
    )
    if combined_value:
        return matrix_U@matrix_S
    else:
        return matrix_U


def predict(matrix, index: int, number_neighbors: int):
    """method which will return indexes the closest documents"""
    return get_neighbors(index, matrix, number_neighbors)


def get_base_for_matrix(counter_docs, token_number, quantity_threshold_up, quantity_threshold_down, token_quantity):
    """nested method"""
    ar_row = []
    ar_col = []
    ar_val = []
    for idx, item in enumerate(counter_docs):
        needed_items = {token: quantity for token, quantity in item.items() if (token_quantity[token] > quantity_threshold_down) and (token_quantity[token] < quantity_threshold_up)}
        ar_row += [idx] * len(needed_items)
        ar_col += [token_number[token] for token in needed_items.keys()]
        ar_val += list(needed_items.values())
    return ar_row, ar_col, ar_val


def get_neighbors(idxs, matrix, number_of_neighbors):
    """nested method"""
    neighbors_for_doc = {}
    for idx in idxs:
        norms = LA.norm(matrix, axis=1, keepdims=True)
        matrix /= norms
        doc = matrix[idx]
        mult = np.matmul(matrix, doc)
        ar = np.argpartition(a=mult, kth=-number_of_neighbors)
        neighbors_for_doc[idx] = ar[-number_of_neighbors - 1:len(ar) - 1]
    return neighbors_for_doc


def get_document_term_matrix(counter_docs, token_number, quantity_threshold_up, quantity_threshold_down, token_quantity):
    """nested method"""
    ar_row, ar_col, ar_val = get_base_for_matrix(counter_docs, token_number, quantity_threshold_up, quantity_threshold_down, token_quantity)
    sparse_mtx = csr_matrix(
        (ar_val, (ar_row, ar_col)), shape=(len(counter_docs), len(token_number))
    )
    return get_tf_idf_matrix(sparse_mtx)

def get_tf_idf_matrix(matrix):
    """nested method"""
    number_of_words_in_doc = np.matrix(matrix.getnnz(axis=1)).transpose()
    tf_part = matrix.multiply(1 / number_of_words_in_doc)
    number_docs = matrix.shape[0]
    item = matrix.getnnz(axis=0)
    dataframe = tf_part.multiply(np.log(number_docs / item))
    return tf_part.multiply(dataframe)


# def clean_text(corpus: pd.Series):
#     """nested method"""
#     porter_stemmer = PorterStemmer()
#     corpus = corpus.apply(remove_punctuation)
#     corpus = corpus.apply(tokenization)
#     corpus = corpus.apply(remove_stopwords)
#     corpus = corpus.apply(lambda x: [item.lower() for item in x])
#     corpus = corpus.apply(lambda x: [porter_stemmer.stem(word) for word in x])
#     return corpus.to_numpy()[0]


def clean_text(corpus: str):
    """nested method"""
    porter_stemmer = PorterStemmer()
    corpus = remove_punctuation(corpus)
    corpus = tokenization(corpus)
    corpus = remove_stopwords(corpus)
    corpus = [item.lower() for item in corpus]
    corpus = [porter_stemmer.stem(word) for word in corpus]
    return corpus


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
    common_metric = 0
    counter = 0
    for index in index_doc:
        for item in neighbors:
            if dataframe[name_of_target][item] == dataframe[name_of_target][index]:
                counter += 1
        common_metric += counter / number_of_neighbors
    return common_metric / len(index_doc)
