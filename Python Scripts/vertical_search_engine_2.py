# -*- coding: utf-8 -*-
"""Vertical Search Engine 2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H6X9jhKGdf88ywcx2MY-BdqdLZqH8v6q
"""

import pandas as pd
import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from collections import defaultdict
from collections import Counter
from string import punctuation
import math
import numpy as np

pd.get_option("display.max_columns")

# download nltk features and create punctuation list
nltk.download('stopwords')
nltk.download('punkt')
punctuation = list(punctuation)
# add quote sign "``" and "''"
punctuation.extend(["``","''"])

from google.colab import drive
drive.mount('/content/drive')

# function to convert a webpage in txt format into a sequence of stemmed tokens, stored as list in a dictionary

def load_wikipages(directory):
    """
    load webpages from a given directoy path and tokenize+stem each page and stores it in a dicionary.
    - input: directory
    - returns: dictionary containing the tokenized webpages
    """

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    # initialize dictionary
    docs = {}
    doc_id=1
    # iterate over files in that directory
    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split(')')[0])):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f, 'r', encoding='utf-8') as d:
              url = d.readline().strip()
              doc = d.read()
            word_tokens = nltk.word_tokenize(doc)
            tokens_swremoved = [w for w in word_tokens if w.lower() not in stop_words]
            tokens_stemmed = [stemmer.stem(w) for w in tokens_swremoved]
            tokens_puncremoved = [token for token in tokens_stemmed if token not in punctuation]
            docs[(f'd{doc_id}',url)] = tokens_puncremoved   
            doc_id += 1
    
    return docs
            
directory = "/content/drive/MyDrive/SIADS 699/Raw_TXT_Downloads"
docs = load_wikipages(directory)

len(docs)

# function to transform a given query into a list of tokenized+stemmed words

def query_prep(query):
    """converts a given query into a list of tokenized and stemmed words"""

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    word_tokens = nltk.word_tokenize(query)
    tokens_swremoved = [w for w in word_tokens if w.lower() not in stop_words]
    tokens_stemmed = [stemmer.stem(w) for w in tokens_swremoved]
    tokens_puncremoved = [token for token in tokens_stemmed if token not in punctuation]
    return tokens_puncremoved

# function to build the inverted index

def build_inverted_index(docs, min_df=1):
    """
    builds an inverted index for a collection of given webpages
    - input: docs in dictionary format
    - returns: inverted index as dictionary, where the keys are the words from the webpages, and the values 
        are the mapping of the documents to the word. The values are again a dictionary containing the doc ID 
        as keys, and the values are lists in the format e.g. [2, [3, 5]], which means the word from the key of the
        outer dictionary occurs 2 times at position 3 and 5.
    """

    inv_index = defaultdict(lambda: defaultdict(list))

    for doc in docs:
        posting = defaultdict(list)
        for t_idx, t in enumerate(docs[doc]):
            posting[t].append(t_idx)

        for t in posting:
            inv_index[t][doc].extend([len(posting[t]), posting[t]])
    
    # filter out tokens which appear less than min_df
    inv_index = {token: dict(inv_index[token]) for token in inv_index if len(inv_index[token]) >= min_df}
    
    return inv_index

inv_index = build_inverted_index(docs)

len(inv_index)

"""Store inverted index as pickle file"""

import pickle

# save dictionary to pickle file
with open('/content/drive/MyDrive/SIADS 699/Heroku/inv_index.pickle', 'wb') as file:
  pickle.dump(inv_index, file, protocol=pickle.HIGHEST_PROTOCOL)

"""### Retrieval function from 685 class (TF maximum frequency normalization)"""

# function to retrieve and rank the webpages/documents given a query

def retrieve_n_rank_docs(inverted_index, queries, max_docs=-1):
    """
    Retrieve webpages in order of relevance from an inverted index based on a query. The function looks only 
    at terms of the query which are present in the inverted index. The ranking is based on the retrieval function 
    S(D, Q)=sum(TF(w,D)*IDF(w)*QTF(w)), where TF is caclulated using maximum frequency 
    normalization TF(w,D)=0.5+(0.5xc(w,D)/MaxFreq_w(D)), and IDF(w)=1+ln(n/k). The query term frequency (QTF) is just
    the occurence of the word in the query. 
    
    - input: inverted index, query, max_docs (max number of docs to be retrieved)
    - returns the webpages in descending order of relevance
    
    """
    
    ret_docs = None
    
    # YOUR CODE HERE
    
    # dict for collecting maxFreqw(D) for calulation of normlized TF using maximum frequency normalization, 
    # traverse through all documents in the given inverted index and search for maximum frequency of a word in a document
    max_freq_wD_dict = {}
    for token, posting in inverted_index.items():
        for key, val in posting.items():
            if key not in max_freq_wD_dict:
                max_freq_wD_dict[key] = val[0]
            elif val[0] > max_freq_wD_dict[key]:
                max_freq_wD_dict[key] = val[0]
                
    ret_docs= {}

    for query in queries:
        # dict for each query (temporary)
        docs = {}
        for token in queries[query]:
            # check if token is in index
            if token in inverted_index.keys():
                # IDF: formula: 1 + ln(N/df(w)), Note: IDF is not dependent on a particular document!
                idf = 1 + math.log(len(max_freq_wD_dict) / len(inverted_index[token]))
                # traverse through the documents in the inverted index for the given token, calculate TF*IDF and
                # store the per-document accumulated version of it in dict docs
                # (dictionary accumulating pattern)
                for doc in inverted_index[token]:
                    # calculation of TF*IDF (with normalized version of TF using maxFreqw(D) from dict max_freq_wD_dict)
                    c_wD = inverted_index[token][doc][0]
                    max_freq_wD = max_freq_wD_dict[doc]
                    tfidf = (0.5 + (0.5 * c_wD / max_freq_wD)) * idf 
                    if doc not in docs:
                        docs[doc] = tfidf
                    else:
                        docs[doc] += tfidf
                        
        # rounding before sorting
        docs = {k: round(v,3) for k, v in docs.items()}

        # options for max_docs
        if max_docs == -1:
            docs_per_query = sorted(Counter(docs).most_common(), key=lambda x: (-x[1], int(x[0][0].split('d')[1])))
        else:
            docs_per_query = sorted(Counter(docs).most_common(), key=lambda x: (-x[1], int(x[0][0].split('d')[1])))[:max_docs]
        # grab the doc id out of docs_per_query
        ret_docs[query] = [doc for doc, freq in docs_per_query]
            
#     raise NotImplementedError()
    
    return ret_docs

query = 'relationship, between leukocytes. and cancer:'

queries = {'q': query_prep(query)}
queries

doc_dict = retrieve_n_rank_docs(inv_index, queries, max_docs=20)
# doc_list = [item for sublist in doc_dict.values() for item in sublist]
# docs = [x[1] for x in doc_list]
# set(docs)

# remove duplicated URLs
docs_url = ([url for d_id, url in doc_dict['q']])
sorted(set(docs_url), key=docs_url.index)

"""### Okapi/BM25 Retrieval function (Robertson et al., 99)"""

# OkapiBM25 version of the retrieval function
def OkapiBM25(inverted_index, queries, max_docs=-1, k1=1.2, b=0.75, k3=1000):
    """
    Retrieve webpages in order of relevance from an inverted index based on a query. The function looks only 
    at terms of the query which are present in the inverted index. The ranking is based on the retrieval function 
    S(D, Q)=sum(((k1+1)*c_wD)/(k1*(1-b+b*(|D|/avg_dl)+c_wD)) * ln((N-df(w)+0.5)/(df(w)+0.5) * ((k3+1)*c_wQ)/(k3+c_wQ)). 
    
    - input: inverted index, query, max_docs (max number of docs to be retrieved), and parameters k1, b, k3
    - returns the webpages in descending order of relevance
    
    """
    
    ret_docs = None
       
    # dict for storing the document length
    doc_dict = {}
    for token, posting in inverted_index.items():
        for doc, val in posting.items():
            if doc not in doc_dict:
                doc_dict[doc] = val[0]
            else:
                doc_dict[doc] += val[0]
    # average doc length
    avg_dl = np.mean([value for key, value in doc_dict.items()]) 

    ret_docs= {}            

    for query in queries:
        # dict for each query (temporary)
        docs = {}
        for token in queries[query]:
            # calculate query term frequency c_wQ
            c_wQ = queries[query].count(token)
            # check if token is in index
            if token in inverted_index.keys():
                # IDF: formula: ln((N-df(w)+0.5)/(df(w)+0.5), Note: IDF is not dependent on a particular document!
                idf = math.log((len(doc_dict)-len(inverted_index[token])+0.5)/(len(inverted_index[token])+0.5))
                # traverse through the documents in the inverted index for the given token, calculate the score per token 
                # using Okapi/BM25 and store the per-document accumulated version of it in dict docs
                # (dictionary accumulating pattern)
                for doc in inverted_index[token]:
                    # TF=((k1+1)*c_wD)/(k1*(1-b+b*(|D|/avg_dl)+c_wD))
                    c_wD = inverted_index[token][doc][0]
                    tf = ((k1 + 1) * c_wD) / (k1 * (1 - b + b * (doc_dict[doc]/avg_dl) + c_wD))
                    # normalized query term frequency: qtf=((k3+1)*c_wQ)/(k3+c_wQ)
                    qtf = ((k3 + 1) * c_wQ) / (k3 + c_wQ)
                    score = tf * idf * qtf
                    if doc not in docs:
                        docs[doc] = score
                    else:
                        docs[doc] += score
                        
        # rounding before sorting
        docs = {k: round(v,3) for k, v in docs.items()}

        # options for max_docs
        if max_docs == -1:
            docs_per_query = sorted(Counter(docs).most_common(), key=lambda x: (-x[1], int(x[0][0].split('d')[1])))
        else:
            docs_per_query = sorted(Counter(docs).most_common(), key=lambda x: (-x[1], int(x[0][0].split('d')[1])))[:max_docs]
        # grab the doc id out of docs_per_query
        ret_docs[query] = [(doc,freq) for doc, freq in docs_per_query]
                
    return ret_docs

query = 'is my weight healthy'
query2 = 'what are you going to have for dinner'

queries = {'q': query_prep(query), 'q1':query_prep(query2), }
queries

OkapiBM25(inv_index, queries, max_docs=10, k1=1.2, b=0.75, k3=1000)

"""## With rank-bm25 library"""

pip install rank-bm25

from rank_bm25 import BM25Okapi
import numpy as np

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "windy London"
tokenized_query = query.split(" ")

bm25.get_scores(tokenized_query)
bm25.get_top_n(tokenized_query, corpus, n=3)

docs_list = [d for d in docs.values()]
bm25 = BM25Okapi(docs_list)
query = queries['q']
scores = bm25.get_scores(query)
np.argsort(-scores)+1

(bm25.get_top_n(query, docs_list, n=10))[0]

