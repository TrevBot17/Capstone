import pandas as pd
import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
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
                title = d.readline().strip()
                doc = d.read()
               
            word_tokens = nltk.word_tokenize(doc)
            tokens_swremoved = [w for w in word_tokens if w.lower() not in stop_words]
            tokens_stemmed = [stemmer.stem(w) for w in tokens_swremoved]
            tokens_puncremoved = [token for token in tokens_stemmed if token not in punctuation]
            docs[(f'd{doc_id}',url, title)] = tokens_puncremoved   
            doc_id += 1
    
    return docs
            
directory = "C:/Users/JWeinstein/Capstone-main/src/Raw_TXT_Downloads/"
docs = load_wikipages(directory)

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
with open('C:/Users/JWeinstein/Capstone-main/src/inv_index.pickle', 'wb') as file:
    pickle.dump(inv_index, file, protocol=pickle.HIGHEST_PROTOCOL)
