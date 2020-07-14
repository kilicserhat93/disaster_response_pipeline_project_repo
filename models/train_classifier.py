# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pickle
import sys


def load_data(database_filepath):
    engine = create_engine("sqlite:///"+database_filepath)
    #engine = create_engine('sqlite:///disaster_db.db')
    df = pd.read_sql_table("disaster_table", engine)
    X = df.message
    y = df.drop(columns = ["genre","message","id", "original"], axis=1)
    return X, y, y.columns


def tokenize(text):
    return word_tokenize(text)


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3), n_jobs=-1))
])
    param_grid2 = dict(clf__estimator__n_neighbors = [2,3,4])
    grid_search_model = GridSearchCV(estimator=pipeline, param_grid=param_grid2, cv=2, n_jobs=-1, verbose=2, return_train_score=True)
    return grid_search_model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    for col in category_names:
        print(col ,": \n", classification_report(Y_test[col], Y_pred[col]))
    
    
def save_model(model, model_filepath):
    pickle_filename = "classifier.pkl"  

    with open(pickle_filename, 'wb') as model_file:  
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_path))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()