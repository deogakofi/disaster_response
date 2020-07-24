#import libraries needed
from app import app
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import os
from flask import Flask
import sys
#add database and model paths in the project folder to retrieve model, graphs and data
from pathlib import Path
two_up = Path(__file__).resolve().parents[1]
two_up = str(two_up)
sys.path.append(two_up)

sys.path.append('{}/models'.format(two_up))
import data.figures as f
import pickle
print(pickle.format_version)
#invoke app when script is run



# load data
engine = create_engine("sqlite:////{}/data/emergency.db".format(two_up))
df = pd.read_sql_table('disaster', engine)

def tokenize(text):
    """Tokenizes a df text series

    Args:
        df text series object

    Returns:
        clean token list

    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load model
model = pickle.load(open("{}/models/model_rf_fit.pickle".format(two_up), 'rb'))

@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    """displays cool visuals and receives user input text for model

    Args:
        None

    Returns:
        Plotly  list from data/figures.py and renders the index.html template

    """

    figures = f.return_figures()
    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]
    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)

@app.route('/go')
@app.route('/go.html')
def go():
    """Web page that handles user query and displays model results

    Args:
        None

    Returns:
        query classificaiton results from 'model.pickle'
        and renders the go.html template

    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )
