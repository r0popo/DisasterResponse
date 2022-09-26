import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''IN:
             message_filepath - csv file containing messages sent during the disaster that will be categorised and used in the ML pipeline
            categories_filepath - csv file containing categorisation of all of the received messages across 36 distinct categories
        
       OUT:
             df - dataframe with messages and their respective categories (0 for not related and 1 for related to the category)
    '''
    
    messages =pd.read_csv(messages_filepath, index_col='id') #using id column as index
    categories = pd.read_csv(categories_filepath, index_col='id')
    
    categories = categories['categories'].str.split(";", expand = True) #expanding the categories to multiple columns
    row = categories.iloc[0]
    row.replace('(\-\d)','',regex=True) #extracting only the name of the column form the category string
    category_colnames = row.replace('(\-\d)','',regex=True).tolist()
    categories.columns = category_colnames
    for column in categories: #iteratinf through all of the columns, removing the text part of the string and converting the value to integer
        categories[column] = categories[column].astype(str).apply(lambda x: re.sub('(\D*\-)','',x)).astype('int64')
    
    df = messages.merge(categories, on='id')
    return df

    


def clean_data(df):
    '''IN: 
            df - dataframe with messages and their respective categories (0 for not related and 1 for related to the category)
       OUT: 
            df - cleaned dataframe ready to be used in the ML pipline
    '''
    df.drop_duplicates(subset=['message'], inplace=True)
    return df

    
    
    

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)  
    df.to_sql('DisasterMessages', engine, index=False)
    return
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()