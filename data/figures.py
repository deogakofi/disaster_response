import pandas as pd
import plotly.colors
import plotly.graph_objs as go
from sqlalchemy import create_engine
import os


def return_figures():
    print(".......------------...........------------")
    print(os.getcwd())
    engine = create_engine("sqlite:///../data/figures.db")
    print(engine)
    df = pd.read_sql_table('disaster', engine)
    print(df.columns)






# extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graph_one = []
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending = False)

    genre_counts = pd.DataFrame({'genre':genre_counts.index, 'count':genre_counts.values})


    for genre in genre_counts['genre']:
        x_val = genre_counts[genre_counts['genre'] == genre]['genre'].tolist()
        y_val = genre_counts[genre_counts['genre'] == genre]['count'].tolist()
        graph_one.append(
            go.Bar(
                x=x_val,
                y=y_val,
                marker = dict(color = genre),
                name = genre,
                showlegend = True
            )
        )
        layout_one = dict(title = "Message Genre Counts in Dataset",
                    xaxis = dict(title = 'Genre',
                      autotick=True),
                    yaxis = dict(title = 'Number of Messages'),
                    )

    def count_col():

        c = df.drop(df.columns[:4], axis=1)
        print(c.columns)
        c['related'] = c['related'].astype('str').str.replace('2', '1')
        c['related'] = c['related'].astype('int')
        count = c.sum().sort_values(ascending = False)
        return count
    print(count_col())



    graph_two = []
    pop_cols = count_col().head(10)
    pop_cols_df = pd.DataFrame({'category':pop_cols.index, 'count':pop_cols.values})
    for cat in pop_cols_df['category']:
        x_val = pop_cols_df[pop_cols_df['category'] == cat]['category'].tolist()
        y_val = pop_cols_df[pop_cols_df['category'] == cat]['count'].tolist()
        graph_two.append(
            go.Bar(
                x=x_val,
                y=y_val,
                marker = dict(color = cat),
                name = cat,
                showlegend = True
            )
        )
        layout_two = dict(title = "Popular Message Category in Dataset",
                    xaxis = dict(title = 'Category',
                      autotick=True),
                    yaxis = dict(title = 'Number of Messages'),
                    )




    graph_three = []
    unpop_cols = count_col().tail(10)
    unpop_cols_df = pd.DataFrame({'category':unpop_cols.index, 'count':unpop_cols.values})


    for cat in unpop_cols_df['category']:
        x_val = unpop_cols_df[unpop_cols_df['category'] == cat]['category'].tolist()
        y_val = unpop_cols_df[unpop_cols_df['category'] == cat]['count'].tolist()
        graph_three.append(
            go.Bar(
                x=x_val,
                y=y_val,
                marker = dict(color = cat),
                name = cat,
                showlegend = True
            )
        )
        layout_three = dict(title = "Unpopular Message Category in Dataset",
                    xaxis = dict(title = 'Category',
                      autotick=True),
                    yaxis = dict(title = 'Number of Messages'),
                    )



    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))


    return figures

print(return_figures())
