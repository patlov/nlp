
import pandas as pd
import sqlite3


def main():
    con = sqlite3.connect('dataset/corpus.sqlite3')


    df = pd.read_sql_query("SELECT * FROM Articles", con)
    print(df)

if __name__ == "__main__":
    main()
