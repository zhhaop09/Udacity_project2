import sys
import pandas as pd
import sqlalchemy 

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(f'{messages_filepath}')
    categories = pd.read_csv(f'{categories_filepath}')
    df = pd.merge(messages, categories, on = 'id', how = 'left')
    categories = df['categories'].str.split(";",expand=True)
    row = categories.loc[1,:]
    a = []
    for i in row:
        a.append(i[:-2])
    category_colnames = a
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
        
    df =df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories.reindex(df.index)], axis=1)
    df = df.drop_duplicates()
    return df


def clean_data(df):
    return df
   


def save_data(df, database_filename):
    engine = sqlalchemy.create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messagedata', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

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