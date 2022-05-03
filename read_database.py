import pandas as pd
import sqlite3


# GOAL: try to identify specific posters on their writing style (or additional metadata)


def mergeDF(articles, posts):

    pass


def main():
    con = sqlite3.connect('dataset/corpus.sqlite3')

    articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    posts_df = pd.read_sql_query("SELECT * FROM Posts", con)
    newspaper_staff_df = pd.read_sql_query("SELECT * FROM Newspaper_Staff", con)
    annotations_df = pd.read_sql_query("SELECT * FROM Annotations", con)
    annotations_consolidated_df = pd.read_sql_query("SELECT * FROM Annotations_consolidated", con)
    cross_val_split_df = pd.read_sql_query("SELECT * FROM CrossValSplit", con)
    categories_df = pd.read_sql_query("SELECT * FROM Categories", con)


    print(articles_df)
    print(posts_df)
    print(newspaper_staff_df)

    full_df = mergeDF(articles_df, posts_df)




if __name__ == "__main__":
    main()
