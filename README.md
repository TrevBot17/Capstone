# Team Dream Search Engine
Follow [this link](https://searchenginecapstone.herokuapp.com/) to beta test the search engine for yourself, making sure to submit queries that pertain to the topic of health.

**DISCLAIMER: The information on this site is not intended or implied to be a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health care provider with any questions you may have regarding a medical condition or treatment and before undertaking a new health care regimen, and never disregard professional medical advice or delay in seeking it because of something you have read on this website.**

# Data Flow Architecture Diagram

![](https://i.ibb.co/5GX09tP/Capstone-Architecture-Diagram-3.jpg?raw=True)

# Project Overview
This project offers users a Flask-leveraged web-based search engine to search for Wikipedia articles relating to the topic of health. The search engine queries our data set, which is an inverted index built from a corpus of ~500 Wikipedia articles. The documents are returned based on relevance to the query using the Okapi BM25 algorithm. Additionally, text summaries of each article are included in the search results as well as pagination for user-friendly browsing.

# Jupyter Notebooks
We leveraged several Jupyter Notebooks to execute multiple different functions necessary for the search engine to run. First, the `focused_crawler.ipynb` sources Wikipedia articles based on a seed URL and keyword. For our seed URL, we used the [Wikipedia Article for Health](https://en.wikipedia.org/wiki/Health), and the keyword we used was "health." Next, we set the depth parameter to 10 so that the crawler would go 10 levels deep in Wikipedia space by adding links to articles within the articles that contained the word "health," through 10 iterations of crawling.

The other key notebooks include:


`duplicate_removal.ipynb`: Removes articles that are either identical or different URLs that redirect to the same or nearly identical articles.


`inv_index.ipynb`: Builds inverted index from corpus generated by the Focused Crawler.


`text_summarizer.ipynb`: Provides most salient sentences in summary form from each URL present in the corpus.


`app.ipynb`: Employs Flask library to generate search engine webpage, takes in user query and runs it through the Search Engine (`search_engine.py`), and contains important logic for appending the correct text summarizies to each URL result, as well as pagination information for use in the index.html file. Each result page should only contain 10 results, and the user should be able to proceed through pages individually until they reach the termination of the search results.


`Ground Truth Preparation.ipynb`: Prepares the ground truth table which can be downloaded as Excel sheet to be filled out by the user.


`Evaluation.ipynb`: Evaluates our search engine with a given ground truth sheet.


`WordCloud.ipynb`: Creates a word cloud from all documents in our corpus.

# HTML Base and Index Files
The `index.html` and `base.html` files provide the back-end structure of the search engine webpage. The `index.html` file contains [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) scripting to incorporate logic into the web page's layout dependent on certain conditions. For example, if a user reaches the last page of a certain query's results, they should not be able to click on the next page button since there wouldn't be a next page. Another instance of Jinja2 logic is modifying what is shown on the home page compared to post-search. The user does not need to see an empty section labeled with "No Search Results" on the homepage before they have even executed a query. The `base.html` file contains HTML code for visual aspects like font color and so forth.

# Integration with IDE and Heroku
To make all these scripts work together, we first ran locally the `focused_crawler`, `inv_index`, and `text_summarizer` Jupyter Notebooks to generate pickle files of the inverted index and text summaries of all the documents in the corpus. Make sure to write the pickle to the `app` directory. These pickle files are available pre-generated in this repository at `inv_index.pickle` and `text_summaries.pickle` in `src/app/`. Next, those pickle files are accessed by the `app.py` file (or `app.ipynb` if you would like to run it in a Notebook) in Visual Studio Code or the directory in the same directory as the pickle files, which also contains the `search_engine.py` file along with the HTML files. `app.ipynb` and `search_engine.py` are available here as Jupyter Notebooks. Finally, a Heroku account was set up and the webpage was generated via the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and deployed with Heroku Git. It is recommended that, to successfully run this project, an IDE comparable to VS Code is used following the above-described methodology. To just run the project locally without Heroku deployment, everything can be run in a Jupyter Notebook-style environment.


# How To Run
The first thing you need to do is to download a copy of this project and install the requirements:
```
git clone https://github.com/TrevBot17/Capstone.git
```
```
pip install -r requirements.txt
```
The project includes the following folders with corresponding Jupyter Notebooks and Python scripts: `src/Pre-Processing` and `src/App`. The fully functional webpage can be run by just using the `app.py` (or `app.ipynb`) in the `App` folder. However, if you want to run every from scratch, you'll need to follow the steps below. Note that you'll need to make note of the directory of the project and modify the variable called `working_dir` to the correct location in the notebooks `inv_index.ipynb`and `text_summarizer.ipynb`. This ensures that writing of the pickle files to the `App` folder performs correctly.
## Pre-processing


1. Run the `focused_crawler.ipynb` notebook to generate the corpus. Within this file, you can play around with the parameters like `keyword` and `depth` to modify the content of the corpus. This notebook will create the `Raw_TXT_Downloads` folder that will need to be referenced in future notebooks.
2. Run  `duplicate_removal.ipynb` file to remove articles that are either identical or different URLs that redirect to the same or nearly identical articles. 
3. Run the `inv_index.ipynb` notebook to build the inverted index from your previously generated corpus. Make sure that the `working_dir` variable is set to current location of the project. This file should output the inverted index of the corpus as a pickle file to the `App` folder.
4. Run the `text_summarizer.ipynb` notebook to create text summaries associated with each URL in the corpus. Again, make sure that the `working_dir` variable is set to correct location. This file should output the text summaries of the corpus as a pickle file to the `App` folder.

## App


Run `app.py` (or `app.ipynb`) to build the search engine webpage running on your local machine. This Notebook interacts with the `search_engine.py` script, so make sure to have that script in the same directory.

## Evaluation

In order to evaluate the performance of our search engine run the `Evaluation.ipynb` notebook located in the main `src`folder. This notebook will leverage the existing ground truth Excel sheet `Capstone Ground Truth.xlsx` located in the same folder. If you want to create your own ground truth, you can use the notebook `Ground Truth Preparation.ipynb´, which will create a Pandas DataFrame given a set of queries, which then can be downloaded and filled out by a user in the same way as it was done in the given Excel sheet `Capstone Ground Truth.xlsx`.
