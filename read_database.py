import pandas as pd
import sqlite3
from User import User

# GOAL: try to identify specific posters on their writing style (or additional metadata)


def mergeDF(articles, posts):
    raise NotImplementedError()
    pass




'''
    converting the posts to User objects with the comment
    maybe addition attributes can be helpful at user-class
'''
def convertToUsersWithPosts(posts):
    users = {}

    for key, row in posts.iterrows():
        user_id = row['ID_User']
        if user_id in users:
            current_user = users[user_id]
        else:
            current_user = User(user_id, [])
            users[user_id] = current_user

        current_user.addComment(row['Body'])

    return users



def main():
    con = sqlite3.connect('dataset/corpus.sqlite3')

    articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    posts_df = pd.read_sql_query("SELECT * FROM Posts", con)

    convertToUsersWithPosts(posts_df)


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
