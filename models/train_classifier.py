import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data(database_filepath):
    
    '''
    load data
    Load data from database
    
    Input: 
    database_filepath: filepath to database
    
    Output:
    X: variables
    y: labels
    y.columns: category_names 
    '''
   
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messagedata', engine)
    X = df['message']
    y = df[['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]
    return X, y, y.columns;


def tokenize(text):
    '''
    tokenize
    Tokenize text 
    
    Input: 
    text: text to tokenize
    
    Output:
    clean_tokens : cleaned tockens
    '''
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    build a ml pipeline 
  
    Output:
    pipeline : ML pipeline 
    '''
        
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
     ('mocf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'mocf__estimator': [RandomForestClassifier(), KNeighborsClassifier()]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluate a ml pipeline 
   
    Input:
    model: ml model
    X_test: testing set variables
    Y_test: testing set labels 
    category_names: name of the category 
    
    Output:
    print the classification report for each category 
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.values[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    save_model
    save a ml model to pickle file
   
    Input:
    model: ml model
    model_filepath: filepath to ml model 
    
    '''
    pickle.dump(model, open('models/classifier.pkl', 'wb'))


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

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
