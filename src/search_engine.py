import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

from collections import Counter
import math
import numpy as np
import text_summarizer as text

# function to transform a given query into a list of tokenized+stemmed words
def query_prep(query):
    """convert a given query into a list of tokenized and stemmed words"""
    
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    word_tokens = nltk.word_tokenize(query)
    tokens_swremoved = [w for w in word_tokens if w.lower() not in stop_words]
    tokens_stemmed = [stemmer.stem(w) for w in tokens_swremoved]

    return tokens_stemmed


# OkapiBM25 retrieval function
def OkapiBM25(inverted_index, queries, max_docs=-1, k1=1.2, b=0.75, k3=1000):
    """
    Retrieve webpages in order of relevance from an inverted index based on a query. The function looks only 
    at terms of the query which are present in the inverted index. The ranking is based on the retrieval function 
    S(D, Q)=sum(((k1+1)*c_wD)/(k1*(1-b+b*(|D|/avg_dl))+c_wD) * ln((N-df(w)+0.5)/(df(w)+0.5) * ((k3+1)*c_wQ)/(k3+c_wQ)). 
    
    - input: inverted index, query, max_docs (max number of docs to be retrieved), and parameters k1, b, k3
    - returns the webpages in descending order of relevance
    
    """
    
    ret_docs = None
    
    # YOUR CODE HERE
    
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
                    tf = ((k1 + 1) * c_wD) / (k1 * (1 - b + b * (doc_dict[doc]/avg_dl)) + c_wD)
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
        ret_docs[query] = [doc for doc, freq in docs_per_query]
    
    return ret_docs

