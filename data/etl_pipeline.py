#import modules
import process_data
import pandas as pd

#load data from csv files using process_data.py methods
data = process_data.load_data('disaster_messages.csv', 'disaster_categories.csv')
#creating a separate clean dataset to train model using the process_data.py method
data_clean = process_data.clean_data(data)

#saving a sqlite db for models using the processed data and process_data.py methods
process_data.save_data(data_clean, 'emergency')

def custom_clean_data(df):
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

    return df
#creating a separate clean database to plot figures using the process_data.py method
data = custom_clean_data(data)
process_data.save_data(data, 'figures')
