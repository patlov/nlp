import pandas as pd
import sqlite3


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




if __name__ == "__main__":
    main()
