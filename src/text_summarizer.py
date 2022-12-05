import re
import nltk
import os
import heapq

# download nltk features and create punctuation list
nltk.download('stopwords')
nltk.download('punkt')

def summarizer(text):
    article_text = re.sub(r'\[[0-9]*\]', ' ', text)
    article_text = re.sub(r'\s+', ' ', article_text)
  # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
        maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
        sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)

    summary = ' ... '.join(summary_sentences)
    return summary

directory = 'C:/Users/JWeinstein/Capstone-main/src/Raw_TXT_Downloads/'

dict_summarizer = {}

for filename in sorted(os.listdir(directory), key=lambda x: int(x.split(')')[0])):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        with open(f, 'r', encoding="utf8") as file:
            url = file.readline().strip()
            title = file.readline().strip()
            text = file.read()

        if len(text)>0:
            summary = summarizer(text)
            dict_summarizer[url] = summary

import pickle

# save dict_summarizer to pickle file
with open('[your working directory]/src/text_summaries.pickle', 'wb') as file:
    pickle.dump(dict_summarizer, file, protocol=pickle.HIGHEST_PROTOCOL)

with open("[yor working directory]/src/text_summaries.pickle", "rb") as file:
    text_sum = pickle.load(file)
