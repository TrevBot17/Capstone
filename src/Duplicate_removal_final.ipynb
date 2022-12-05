import os
import nltk
import time

working_dir = 'C:/Users/JWeinstein/Capstone-main/src/'

def jaccard_similarity(list_x, list_y):
    'get jaccard similartiy of two documents'
    
    set_x = set(list_x)
    set_y = set(list_y)
    intersection = set_x.intersection(set_y)
    union = set_x.union(set_y)
    
    return len(intersection) / len(union) if len(union) > 0 else 0

def shingling_jaccard_similarity(text_x, text_y, n):
    'calculate similarity score between two documents using the shingling approach'
    
    x_ngrams=list(nltk.ngrams(text_x.split(" "),n))      
    y_ngrams=list(nltk.ngrams(text_y.split(" "),n))    
    # call function jaccard_similarity 
    sim_score=jaccard_similarity(x_ngrams,y_ngrams)
    
    return sim_score

# code adapted from https://dida.do/blog/how-to-identify-duplicate-files-with-python

directory = f"{working_dir}Raw_TXT_Downloads"
files = sorted(os.listdir(directory), key=lambda x: int(x.split(')')[0]))

# initialize list containing the classes of documents with the same content
duplicates = []

# get start time
startTime = time.time()

# comparison of the documents - nested loop approach
# loop through all files in given path
for file_name in files:
    # open document and get the text
    f = os.path.join(directory, file_name)
    with open(f, 'r', encoding="utf8") as file:
        url = file.readline().strip()
        title = file.readline().strip()
        text = file.read()
    
    is_duplicate = False
    
    # loop through list duplicates, calculate shingling_jaccard_similarity and check if it is duplicate by comparing 
    # the shingling_jaccard_similarity against a threshold of 0.95
    for class_ in duplicates:
        # open first document in class_ in list duplicates
        f = os.path.join(directory, class_[0])
        with open(f, 'r', encoding="utf8") as file:
            url = file.readline().strip()
            title = file.readline().strip()
            class_text = file.read()     
        
        # call function shingling_jaccard_similarity, with trigrams (n=3)
        shing_jaccard = shingling_jaccard_similarity(text, class_text, 3)
        
        if shing_jaccard > 0.95:
            is_duplicate = True
        else:
            is_duplicate = False

        if is_duplicate:
            class_.append(file_name)            
            break
    
    if not is_duplicate:
        duplicates.append([file_name])     


# remove the duplicates from directory: only keep the first document within the inner lists 
for class_ in duplicates:
    for file in class_[1:]:
        os.remove(os.path.join(directory, file))
