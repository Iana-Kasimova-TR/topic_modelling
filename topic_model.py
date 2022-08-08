from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import neighbors
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.stem.porter import PorterStemmer
from numpy import linalg as LA
import json
import re
import string
import nltk
from nltk.corpus import stopwords

# fit(return state) and predict(use twice) separately 
# sent path to file and iterate, methods accept generators
# for demo not needed all dataset (slice), polars(replacement for pandas)

def fit(path_to_file: str, name_text_column: str):
    token_number = {}
    token_quantity = {}
    counters_docs = []
    for line in pd.read_csv(path_to_file, chunksize=1, sep=','):
        line = clean_text(line[name_text_column])
        token_number, token_quantity = get_token_dictionaries(line, token_number, token_quantity)
        counters_docs.append(Counter(line))
    document_term_mtx = get_document_term_matrix(counters_docs, token_number)    
    U, _, _ = randomized_svd(document_term_mtx, n_components=5, n_iter=5, random_state=None)
    return U   


def predict(U, index: int, number_neighbors: int):
    return get_neighbors(index, U, number_neighbors)    


def get_base_for_matrix(counter_docs, token_number):
    ar_row = []
    ar_col = []
    ar_val = []
    for idx, item in enumerate(counter_docs):
        ar_row.extend(np.repeat(idx, len(item)))
        ar_col.extend([token_number.get(token) for token in item.keys()])
        ar_val.extend([value for value in item.values()])
    return  ar_row, ar_col, ar_val

def get_token_dictionaries(line, token_number, token_quantity):
    counter = 0
    for word_item in line:
        if(word_item not in token_number.keys()):
            token_number[word_item] = counter
            counter += 1
        token_quantity[word_item] =  token_quantity.get(word_item, 0) + 1        
    return token_number, token_quantity    


# use cos distance, consider 10 docs(query) 1*256 - doc on 256*90 - docs, sort, avrage precision at key.
# each row in matrix divide on ||x|| = sqrt(sum(x_i^2)), where x it is row
def get_neighbors(idx, U, k):    
    norms =  LA.norm(U, axis = 1, keepdims = True)   
    U_norm /= norms        
    doc = U_norm[idx]
    mult = np.matmul(U_norm, doc)
    return np.argpartition(a=mult, kth=-k, axis=1)[-k:]


def get_document_term_matrix(counter_docs, token_number):
    ar_row, ar_col, ar_val = get_base_for_matrix(counter_docs, token_number)
    sparse_mtx = csr_matrix((ar_val, (ar_row, ar_col)), shape = (len(counter_docs), len(token_number)))
    return get_tf_idf_matrix(sparse_mtx)

def get_tokens_from_doc(doc):
    # if doc is a list of tokens
    # can we iterate by set
    # constructor for counter, keys - id of tokens - list 
    return set(doc)

def count_quantity_token_in_doc(doc, token):
    counter = 0
    for item in doc:
        if item == token:
            counter +=1
    return counter           

def get_tf_idf_matrix(mx):
    number_of_words_in_doc = np.matrix(mx.getnnz(axis=1)).transpose()
    print(number_of_words_in_doc)
    tf = mx.multiply(1/number_of_words_in_doc)
    number_docs = mx.shape[0]
    item = mx.getnnz(axis=0)
    df = tf.multiply(np.log(number_docs/item))
    return tf.multiply(df)       

def clean_text(corpus: pd.Series):
    porter_stemmer = PorterStemmer()
    corpus = corpus.apply(lambda x: remove_punctuation(x))
    corpus = corpus.apply(lambda x: tokenization(x))
    corpus = corpus.apply(lambda x: remove_stopwords(x))
    corpus = corpus.apply(lambda x: [item.lower() for item in x])
    corpus = corpus.apply(lambda x: [porter_stemmer.stem(word) for word in x])
    return corpus.to_numpy()[0]


def remove_punctuation(text):
    punctuation_free="".join([i for i in text if i not in string.punctuation])
    return punctuation_free


def tokenization(text):
    tokens = re.split(r'\W+',text)
    return tokens   


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords.words('english')]
    return output


#use cos distance, consider 10 docs(query) 1*256 - doc on 256*90 - docs, sort, avrage precision at key.
#each row in matrix divide on ||x|| = sqrt(sum(x_i^2)), where x it is row
def get_neighbors(idx, u, k):
    doc = u[idx]
    docs = np.vstack((u[:idx, :], u[idx+1:,]))
    mult = np.matmul(docs, doc)
    return np.argsort(mult)[-k:]    


def get_correctness_doc_count(df, index_doc, u, k):
    neighbors = get_neighbors(index_doc, u, k)
    counter = 0
    for item in neighbors:
        if(df.Category[item] == df.Category[index_doc]) :
            counter +=1  
    return counter/k         