# import libraries
import nltk
import re
nltk.download('punkt')
nltk.download('wordnet')
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import gmean
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
import pickle




def load_data(engine_name, table_name):
    # load data from database
    engine = create_engine('sqlite:///../data/{}.db'.format(engine_name))
    df = pd.read_sql("SELECT * FROM {}".format(table_name), engine)
    X = df['message']
    Y = df
    Y = Y.drop(Y.columns[:3], axis=1)
    Y= Y.astype(int)
    return X, Y



def tokenize(df_series):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = []
    for tok in df_series:
        clean = tokenizer.tokenize(tok)
        tokens.append(clean)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(str(tok)).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(classifier):
    pipeline = Pipeline(
        [('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(classifier))]
        )

    return pipeline



def train_model(X, Y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model_fit = model.fit(X_train.values, y_train.values)
    y_pred = model_fit.predict(X_test)


    return X_train, X_test, y_train, y_test, y_pred, model

def get_results(y_test, y_pred):
    results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f_score'])
    count = 0
    for category in y_test.columns:
        precision, recall, f_score, support = score(y_test[category], y_pred[:,count], average='weighted')
        results.at[count+1, 'category'] =category
        results.at[count+1, 'precision'] = precision
        results.at[count+1, 'recall'] = recall
        results.at[count+1, 'f_score'] = f_score
        count += 1
    avg_precision = results['precision'].mean()
    print('Average precision:', avg_precision)
    print('Average recall:', results['recall'].mean())
    print('Average f_score:', results['f_score'].mean())
    return results



def grid_search(model, X_train, y_train):

    param = {
            'clf__estimator__n_estimators': [100, 200],
            'vect__max_df': (0.5, 0.75, 1.0)
        }

    cv = GridSearchCV(model, param_grid=param, verbose = 2, n_jobs = -1)
    cv.fit(X_train.values, y_train.values)

    print("\nBest Parameters_rf:", cv.best_params_)
    print("Best cross-validation_rf score: {:.2f}".format(cv.best_score_))
    print("Best cross-validation score_rf: {}".format(cv.cv_results_))

    return cv

def save_cv(cv_name, cv):
    with open('../models/{}.pickle'.format(cv_name), 'wb') as f:
        pickle.dump(cv, f)

def save_clf_results(results_name, results):
    with open('../models/{}.pickle'.format(results_name), 'wb') as f:
        pickle.dump(results, f)
def save_model(model_name, model):
    with open('../models/{}.pickle'.format(model_name), 'wb') as f:
        pickle.dump(model, f)
