import sys
import pandas as pd
import sqlalchemy 

def load_data(messages_filepath, categories_filepath):
    '''
    load data 
    Load data and merge into a single dataframe
    
    Input:
    messages_filepath: filepath to messages csv file
    categories_filepath: filepath to categories csv file
    
    Output:
    df: dataframe combining messages and categories data
    '''
    messages = pd.read_csv(f'{messages_filepath}')
    categories = pd.read_csv(f'{categories_filepath}')
    df = pd.merge(messages, categories, on = 'id', how = 'left')
    return df


def clean_data(df):
    '''
    clean data 
    Clean data and get ready for ml
    
    Input:
    df: original dataframe
    
    Output:
    df: cleaned dataframe(renamed the columns, transformed the data, and dropped the repeated rows) 
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.loc[1,:]
    a = []
    for i in row:
        a.append(i[:-2])
    category_colnames = a
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: 1 if int(x[-1]) >0 else 0)   
        
    df =df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories.reindex(df.index)], axis=1)
    df = df.drop_duplicates(）
    return df
   

def save_data(df, database_filename):
    '''
    save data 
    Save data to database
    
    Input:
    df: dataframe
    database_filename: filepath to database
    '''
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messagedata', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Clean data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
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
