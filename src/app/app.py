from math import ceil
from flask import Flask, render_template, request
from search_engine import query_prep, OkapiBM25
import pickle
working_dir = 'C:/Users/JWeinstein/Capstone-main'
app = Flask(__name__)
@app.route('/')
def results():
    return render_template('index.html', page = -1)

@app.route('/search/<int:page>', methods=['POST'])
def index(page):
    with open(f"{working_dir}/src/inv_index.pickle", "rb") as file:
        inv_ind = pickle.load(file)
    user_search_query = request.form.get('query')

    
    queries = {'q': query_prep(user_search_query)}
    ranking = OkapiBM25(inv_ind, queries)['q']
    seen = set()

    newRes= []
    myDict = pickle.load(open(f'{working_dir}/src/text_summaries.pickle','rb'))
    
    for r in ranking:
        newRes.append((r[1], r[2], myDict[r[1]]))
   
    start = 0+(page*10)
    end = start + 10
    upper = ceil((len(newRes)/10))

    subset = newRes[start:end]
    if end > len(newRes):
        end = len(newRes)
    return render_template('index.html', search_results_list = subset,
                                          user_query=user_search_query,
                                          lower = 0,
                                          upper = upper,
                                          page = page,
                                          start = start,
                                          end = end,
                                          total = len(newRes))


if __name__ == "__main__":
    app.run()