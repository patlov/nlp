import pandas as pd
import sqlite3
from User import User


# GOAL: try to identify specific posters on their writing style (or additional metadata)

# Step 1: get all Posts from a specific user-id
# Step 2: preprocessing von allen posts von einem user
# Step 3: feature identification (corr matrix)
# Step 4: user identifizieren basierend auf writing style

def mergeDF(articles, posts):
    return pd.merge(articles, posts, on='ID_Article')


'''
    converting the posts to User objects with the comment
    maybe additional attributes can be helpful at user-class
'''


def convertToUsersWithPosts(posts):
    users = {}

    for key, row in posts.iterrows():
        user_id = row['ID_User']
        if user_id in users:
            current_user = users[user_id]
        else:
            current_user = User(user_id, [], -1, -1, '')
            users[user_id] = current_user

        current_user.addComment(row['Body'])
        current_user.setPositiveVotes(row['PositiveVotes'])
        current_user.setNegativeVotes(row['NegativeVotes'])
        current_user.setCreationDate(row['CreatedAt'])

    return users


def main():
    con = sqlite3.connect('dataset/corpus.sqlite3')

    articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    posts_df = pd.read_sql_query("SELECT * FROM Posts", con)

    users = convertToUsersWithPosts(posts_df)

    # newspaper_staff_df = pd.read_sql_query("SELECT * FROM Newspaper_Staff", con)
    # annotations_df = pd.read_sql_query("SELECT * FROM Annotations", con)
    # annotations_consolidated_df = pd.read_sql_query("SELECT * FROM Annotations_consolidated", con)
    # cross_val_split_df = pd.read_sql_query("SELECT * FROM CrossValSplit", con)
    # categories_df = pd.read_sql_query("SELECT * FROM Categories", con)

    print(users)

    full_df = mergeDF(articles_df, posts_df)
    print(full_df)


if __name__ == "__main__":
    main()
