import sys
import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''IN: database_filepath - filepath of the database with data cleaned during the ETL process
       OUT: X - list of message strings
            Y - list of classification categories
            category_colnames - list of category names'''
    
    engine = create_engine('sqlite:///' + database_filepath)
    connection = engine.connect()
    metadata = db.MetaData()
    table = db.Table('DisasterMessages', metadata, autoload=True, autoload_with=engine)

    query = db.select([table]) 
    ResultProxy = connection.execute(query)
    ResultSet = ResultProxy.fetchall()
    df = pd.DataFrame(ResultSet)
    df.columns = ResultSet[0].keys()
    
    category_colnames = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']

    X = df.message
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]
    return X, Y, category_colnames


def tokenize(text):
    ''' IN: text - string
        OUT: clean_tokens - tokenised test using WordNetLemmatizer'''
    tokens = nltk.word_tokenize(text) 
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    clean_tokens = []
    for tok in tokens:
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", tok.lower()).strip()
        clean_tok = WordNetLemmatizer().lemmatize(tok)
        clean_tok = WordNetLemmatizer().lemmatize(clean_tok, pos ='v')
        
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), #using Term Frequency times Inverse Document Frequency
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [50,75],
        'clf__estimator__learning_rate': [0.975,1.0]
            }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=5)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_t=np.transpose(Y_pred)
    
    i=0
    for col in category_names:
        Y_test_col = Y_test[col].values
        Y_pred_col = Y_pred_t[i]
        print (col)
        print(classification_report(Y_test_col, Y_pred_col))
        i+=1
    
    accuracy = (Y_pred == Y_test).mean() 
    print (accuracy)
    return


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    return


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