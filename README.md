# Team Dream Search Engine
Follow [this link](https://searchenginecapstone.herokuapp.com/) to beta test the search engine for yourself.

**DISCLAIMER: The information on this site is not intended or implied to be a substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health care provider with any questions you may have regarding a medical condition or treatment and before undertaking a new health care regimen, and never disregard professional medical advice or delay in seeking it because of something you have read on this website.**

# Data Flow Architecture Diagram

![](https://i.ibb.co/52r2RZk/Capstone-Architecture-Diagram.jpg?raw=True)

# Project Overview
This project offers users a Flask-leveraged web-based search engine to search for Wikipedia articles relating to the topic of health. The search engine queries our data set, which is an inverted index built from a corpus of ~500 Wikipedia articles. The documents are returned based on relevance to the query using the Okapi BM25 algorithm. Additionally, text summaries of each article are included in the search results as well as pagination for user-friendly browsing.

# Python Scripts
We leveraged several Python scripts to execute multiple different functions necessary for the search engine to run. First, the `focused_crawler.py` sources Wikipedia articles based on a seed URL and keyword. For our seed URL, we used the [Wikipedia Article for Health](https://en.wikipedia.org/wiki/Health), and the keyword we used was "health." Next, we set the depth parameter to 10 so that the crawler would go 10 levels deep in Wikipedia space by adding links to articles within the articles that contained the word "health," through 10 iterations of crawling.

The other key Python scripts include:


`inv_index.ipynb`: Builds inverted index from corpus generated by the Focused Crawler


`search_engine.ipynb`: Runs query against inverted index, ranks document results based on Okapi BM25 algorithm, and returns URL results.


`text_summarizer.ipynb`: Provides most salient sentences in summary form from each URL present in the corpus.


`app.ipynb`: Employs Flask library to generate search engine webpage, takes in user query and runs it through the Search Engine Python script, and contains important logic for appending the correct text summarizies to each URL result, as well as pagination information for use in the index.html file. Each result page should only contain 10 results, and the user should be able to proceed through pages individually until they reach the termination of the search results.

# HTML Base and Index Files
The `index.html` and `base.html` files provide the back-end structure of the search engine webpage. The `index.html` file contains [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) scripting to incorporate logic into the web page's layout dependent on certain conditions. For example, if a user reaches the last page of a certain query's results, they should not be able to click on the next page button since there wouldn't be a next page. Another instance of Jinja2 logic is modifying what is shown on the home page compared to post-search. The user does not need to see an empty section labeled with "No Search Results" on the homepage before they have even executed a query. The `base.html` file contains HTML code for visual aspects like font color and so forth.

# Integration with IDE and Heroku
To make all these scripts work together, we first ran locally the `focused_crawler`, `inv_index`, and `text_summarizer` Jupyter Notebooks to generate pickle files of the inverted index and text summaries of all the documents in the corpus and saved those to a local directory. These pickle files are available pre-generated in this repository at `inv_index.pickle` and `text_summaries.pickle` in `src/Pre-Processing`. Next, those pickle files were accessed by the `app.py` file run in Visual Studio Code in the same directory as the pickle files, which also contained the `search_engine.py` file along with the HTML files. `app.py` and `search_engine.py` are available here as Jupyter Notebooks. Finally, a Heroku account was set up and the webpage was generated via the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and deployed with Heroku Git. It is recommended that, to successfully run this project, an IDE comparable to VS Code is used following the above-described methodology. To just run the project locally without Heroku deployment, everything can be run in a Jupyter Notebook-style environment.


# How To Run
Before you can run the fully functional webpage, you'll need to use some of the Python scripts to generate the corpus and text summaries. This way, this long process will only have to be completed once, rather than each time you want to execute a search. To accomplish this, you'll need to take note of the directory in which you'll be running these notebooks and modify all variables called `working_dir` in each notebook to the correct locations so that reading and writing of files/folders performs correctly. Also, make sure to have a folder named `templates` with `index.html` and `base.html`.
## Pre-processing


1. Run the `focused_crawler.ipynb` Notebook to generate the corpus. Within this file, you can play around with the parameters like `keyword` and `depth` to modify the content of the corpus. Make sure to write this file to the same directory with the other Python scripts. This script will creates the  `Raw_TXT_Downloads` folder that will need to be referenced in future scripts.
2. Run  `duplicate_removal.ipynb` file to remove articles that are either identical or different URLs that redirect to the same or nearly identical articles. 
3. Run the `inv_index.ipynb` Notebook to build the inverted index from your previously generated corpus. Again, make sure that the `working_dir` variable is set to your current directory. This file should output the inverted index of the corpus as a pickle file to your working directory. You should also run `text_summaries.ipynb`  to create another pickle file in your working directory containing the text summaries associated with each URL in the corpus. 

## App


5. Run `app.ipynb` to build the search engine webpage running on your local machine. This Notebook interacts with the `search_engine.py` script.
6. Add details about `evaluation` and prec-rec python functions


<!-- Once these steps are complete, you can run the project from your terminal:


Install virtualenv:

`$ pip install virtualenv`

Open a terminal in the project root directory and run:

`$ virtualenv env`

Then run the command:

`$ .\env\Scripts\activate`

Then install the dependencies:

`$ (env) pip install -r requirements.txt`

Finally start the web server:

`$ (env) python app.py`

This server will start on port 5000 by default. You can change this in app.py by changing the following line to this:

```
if __name__ == "__main__":
    app.run(debug=True, port=<desired port>)
``` -->
