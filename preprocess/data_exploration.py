import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

'''
    visualize when (time) comments are written
'''
def showCommentTime(users_df: pd.DataFrame):
    pass


'''
    visualize the number of comments per user
'''
def showNrOfCommentsPerUser(users_df: pd.DataFrame):

    users_df_list = users_df.to_dict('records')
    authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common()
    author_ids,comment_count = zip(*authors)

    plt.title("Number of comments per author")
    plt.boxplot(comment_count)
    plt.xlabel("Authors")
    plt.ylabel("Number of Comments")
    plt.savefig("assets/comments_per_author1.png")

    plt.title("Number of comments per author (zoomed in)")
    plt.boxplot(comment_count)
    plt.ylim(0, 80)
    plt.yticks([y for y in range(0,80,5)])
    plt.xlabel("Authors")
    plt.ylabel("Number of Comments")
    plt.savefig("assets/comments_per_author2.png")
    plt.clf()


def showNrOfCommentsPerUserBarChart(users_df : pd.DataFrame):
    users_df_list = users_df.to_dict('records')
    authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common()
    author_ids,comment_count = zip(*authors)

    plt.clf()
    plt.title("Number of comments per author")
    plt.stackplot([i for i in range(len(author_ids))], comment_count)
    plt.ylim(0, 1000)
    plt.xlim(0,len(comment_count))
    plt.yticks([y for y in range(0,1000,50)])
    plt.xlabel("Authors")
    plt.ylabel("Number of Comments")
    plt.savefig("assets/comments_per_author3.png")

def userStats(df : pd.DataFrame, title : str):
    users_df_list = df.to_dict('records')
    authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common()
    author_ids,comment_count = zip(*authors)

    print("----------------------------------------------------")
    print(title)
    print(f"number of authors {len(author_ids)}")
    print(f"number of comments {len(df)}")
    print(f"max comments of authors {max(comment_count)}")
    print(f"min comments of authors {min(comment_count)}")
    print(f"mean comments of authors {sum(comment_count) / len(comment_count)}")



def getUsersWithMinNumberOfComments(df : pd.DataFrame, min_comments) -> pd.DataFrame:
    users_df_list = df.to_dict('records')

    authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common()
    author_ids,comment_count = zip(*authors)

    # get a list of authors we want to remove
    authors_to_drop = []
    for i, author in enumerate(author_ids):
        if comment_count[i] < min_comments:
            authors_to_drop.append(author)

    # remove the authors from df and return
    df_to_remove = df[df['ID_User'].isin(authors_to_drop)]
    df.drop(df_to_remove.index, inplace=True)
    return df


def preprocessingSteps(users_df : pd.DataFrame):
    # remove none and empty entries
    users_df = users_df.replace(to_replace=['None', ''], value=np.nan).dropna()

    showNrOfCommentsPerUser(users_df)
    userStats(users_df, "All Users")

    users_subset = getUsersWithMinNumberOfComments(users_df, 50)
    userStats(users_subset, "Only relevant users")
    showNrOfCommentsPerUserBarChart(users_subset)
    return users_subset
