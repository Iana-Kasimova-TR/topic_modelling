"""module for keeping logic for topic modelling"""
import csv
import re
import string
from collections import Counter

import numpy as np
from nltk.corpus import stopwords, names
from nltk.stem import WordNetLemmatizer
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def fit(
    path_to_file: str,
    number_text_column: int,
    n_components: int,
    n_iter: int,
    random_state,
    algorithm: str,
    quantity_threshold_up: int,
    quantity_threshold_down: int,
):
    """method which provide document-term matrix"""
    token_number: dict = {}
    token_counter = Counter()
    counters_docs = []
    with open(path_to_file, newline="", encoding="utf-8") as csvfile:
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
    document_term_mtx = get_document_term_matrix(
        counters_docs,
        token_number,
        quantity_threshold_up,
        quantity_threshold_down,
        token_quantity,
    )
    svd = TruncatedSVD(
        n_components=n_components,
        n_iter=n_iter,
        random_state=random_state,
        algorithm=algorithm,
    )
    lsa_matrix = svd.fit_transform(document_term_mtx)
    return lsa_matrix, svd, token_number


def predict(matrix, index: int, number_neighbors: int):
    """method which will return indexes the closest documents"""
    return get_neighbors(index, matrix, number_neighbors)


def get_base_for_matrix(
    counter_docs,
    token_number,
    quantity_threshold_up,
    quantity_threshold_down,
    token_quantity,
):
    """nested method"""
    ar_row = []
    ar_col = []
    ar_val = []
    for idx, item in enumerate(counter_docs):
        needed_items = {
            token: quantity
            for token, quantity in item.items()
            if (token_quantity[token] > quantity_threshold_down)
            and (token_quantity[token] < quantity_threshold_up)
        }
        ar_row += [idx] * len(needed_items)
        ar_col += [token_number[token] for token in needed_items.keys()]
        ar_val += list(needed_items.values())
    return ar_row, ar_col, ar_val


def get_document_term_matrix(
    counter_docs,
    token_number,
    quantity_threshold_up,
    quantity_threshold_down,
    token_quantity,
):
    """nested method"""
    ar_row, ar_col, ar_val = get_base_for_matrix(
        counter_docs,
        token_number,
        quantity_threshold_up,
        quantity_threshold_down,
        token_quantity,
    )
    sparse_mtx = csr_matrix(
        (ar_val, (ar_row, ar_col)), shape=(len(counter_docs), len(token_number))
    )
    return get_tf_idf_matrix(sparse_mtx)


def get_tf_idf_matrix(matrix):
    """nested method"""
    number_of_words_in_doc = np.matrix(matrix.getnnz(axis=1)).transpose()
    tf_part = matrix.multiply(1 / number_of_words_in_doc)
    number_docs = matrix.shape[0]
    # item it is in how many documents we face this token
    item = matrix.getnnz(axis=0)
    dataframe = tf_part.multiply(np.log(number_docs / item))
    return dataframe


def clean_text(corpus: str):
    """nested method"""
    lemmatizer = WordNetLemmatizer()
    corpus = remove_punctuation(corpus)
    corpus = tokenization(corpus)
    corpus = remove_unuseful_words(corpus)
    corpus = remove_common_word(corpus)
    corpus = [item.lower() for item in corpus]
    corpus = [lemmatizer.lemmatize(word) for word in corpus]
    return corpus

def remove_unuseful_words(text):
    result = []
    for t in text:
        if len(t) > 2:
            result.append(t)
    return result        


def remove_punctuation(text):
    """nested method"""
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    return punctuation_free


def tokenization(text):
    """nested method"""
    tokens = re.split(r"\W+", text)
    return tokens


def remove_common_word(text):
    """nested method"""
    sets = [stopwords.words("english"), names.words("male.txt"), names.words("female.txt")]
    return [i for i in text if not any(i in item for item in sets)]


def get_neighbors(idxs, matrix, number_of_neighbors):
    """nested method"""
    neighbors_for_doc = {}
    for idx in idxs:
        norms = LA.norm(matrix, axis=1, keepdims=True)
        matrix /= norms
        doc = matrix[idx]
        mult = np.matmul(matrix, doc)
        array_neighbors = np.argpartition(a=mult, kth=-number_of_neighbors)
        neighbors_for_doc[idx] = array_neighbors[
            -number_of_neighbors - 1 : len(array_neighbors) - 1
        ]
    return neighbors_for_doc


# def get_correctness_doc_count(
#     dataframe, index_doc, matrix, number_of_neighbors, name_of_target
# ):
#     """method wich will return value of metric"""
#     neighbors = get_neighbors(index_doc, matrix, number_of_neighbors)
#     common_metric = 0
#     counter = 0
#     for index in index_doc:
#         for item in neighbors:
#             if dataframe[name_of_target][item] == dataframe[name_of_target][index]:
#                 counter += 1
#         common_metric += counter / number_of_neighbors
#     return common_metric / len(index_doc)


def calc_metric_for_simularity_matrix(simularity_matrix, targett, k):
    """
    calculate metric
    """
    indexes_of_docs = np.argpartition(-simularity_matrix, axis=0, kth=k)
    indexes_of_most_similar_docs = indexes_of_docs[: indexes_of_docs.shape[0], :k]
    compare_group = np.add(targett[indexes_of_most_similar_docs].T, -targett)
    return np.mean(np.count_nonzero(compare_group == 0, axis=0) / k)


def calc_map_metric(k, matrix, targett):
    """
    calculate metric through cosinus calculate similarity
    """
    row_sums = matrix.sum(axis=1)
    norm_dtm_svd_matrix = matrix / row_sums[:, np.newaxis]
    dtm_svd_matrix_transpose = matrix.transpose()
    dot_norm_transpose = norm_dtm_svd_matrix.dot(dtm_svd_matrix_transpose)
    np.fill_diagonal(dot_norm_transpose, -1)
    return calc_metric_for_simularity_matrix(dot_norm_transpose, targett, k)


def get_keywords_for_topic(k, model, dict_token):
    for i, comp in enumerate(model.components_):
        vocab_comp = zip(dict_token, comp)
    sorted_words = sorted(vocab_comp, key=lambda x:x[1], reverse=True)[:k]
    print("\n".join(sorted_words))