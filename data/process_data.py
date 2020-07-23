# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load data from csv files and merge both datasets on ID

    Args:
        messages_filepath => csv file for messages
        categories_filepath => csv file for categories
    Returns:
        df => Dataframe messages and categories merged on ID

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df

# clean categories and merge to messages
def clean_data(df):
    """Clean categories and merge to messages

    Args:
        df => DataFrame of merged categories and messages csv files

    Returns:
        df => Dataframe of cleaned categories and dropped duplicateds

    """
    categories = pd.Series(df.categories).str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:]).values

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    df = pd.concat([df,pd.get_dummies(df['genre'])],axis=1)
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    df = df.drop(['genre','social'],axis=1)

    return df


def save_data(df, database_filename):
    """Save dataframe to sqlite engine

    Args:
        df => DataFrame of merged categories and messages csv files
        database_filename => filename for db engine as string
    Returns:
        None

    """
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('disaster', engine, index=False, if_exists='replace')
