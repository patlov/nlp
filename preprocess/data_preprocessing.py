import logging
import sys

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from enum import Enum
from datetime import datetime
from preprocess.nlp_preprocessing import nlp_preprocess_text

class WritingTime(Enum):
    Morning = 1
    Midday = 2
    Afternoon = 3
    Evening = 4
    Night = 5



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

'''
    visualize the number of comments per user as bar chart
'''
def showNrOfCommentsPerUserBarChart(users_df : pd.DataFrame):
    users_df_list = users_df.to_dict('records')
    authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common()
    author_ids,comment_count = zip(*authors)


    plt.clf()
    plt.hist()

    # plt.clf()
    # plt.title("Number of comments per author")
    # plt.stackplot([i for i in range(len(author_ids))], comment_count)
    # plt.ylim(0, max(comment_count))
    # plt.xlim(0,len(comment_count))
    # plt.xlabel("Authors")
    # plt.ylabel("Number of Comments")
    # plt.savefig("assets/comments_per_author3.png")

'''
    get user stats of the current dataset
'''
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



'''
    remove all users which have less than min_comments of comments
'''
def CutUsersLowerLimit(df : pd.DataFrame, min_comments) -> pd.DataFrame:
    print("Remove users with LESS than " + str(min_comments) + " comments")
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


'''
    only allow max_comments of comments per user
'''
def cutUsersUpperLimit(users : pd.DataFrame, max_comment : int):

    users_comments_count = {}
    reduced_users = []
    for index, row in tqdm(users.iterrows(), total=users.shape[0], desc="Remove users with MORE than " + str(max_comment) + " comments"):

        user_id = row['ID_User']
        if user_id not in users_comments_count:
            users_comments_count[user_id] = 0
        if users_comments_count[user_id] >= max_comment:
            continue
        else:
            reduced_users.append(row)
            users_comments_count[user_id] += 1

    print("converting to dataFrame")
    subset_df = pd.DataFrame(reduced_users)
    subset_df = subset_df.reset_index(drop=True)
    return subset_df


def calculateWritingTime(users : pd.DataFrame, plot):

    writing_time_list = []
    writing_hour_list = []
    for index, row  in users.iterrows():
        date = datetime.strptime(row['CreatedAt'], '%Y-%m-%d %H:%M:%S.%f')
        writing_hour_list.append(date.hour)
        if date.hour >= 6 and date.hour < 11:
            writing_time_list.append(WritingTime.Morning)
        elif date.hour >= 11 and date.hour < 14:
            writing_time_list.append(WritingTime.Midday)
        elif date.hour >= 14 and date.hour < 18:
            writing_time_list.append(WritingTime.Afternoon)
        elif date.hour >= 18 and date.hour < 23:
            writing_time_list.append(WritingTime.Evening)
        elif date.hour == 23 or date.hour < 6:
            writing_time_list.append(WritingTime.Night)
        else:
            assert(str(date) + " date not possible")
    users['WritingTime'] = writing_time_list

    if plot:
        plt.hist(writing_hour_list, bins=24)
        plt.title("Number of comments for each hour")
        plt.xticks([y for y in range(0, 24, 1)])
        plt.xlabel("Hours")
        plt.ylabel("Number of Comments")
        plt.savefig("assets/comments_per_hour.png")
    return users

def doNLPpreprocessing(users : pd.DataFrame):

    for index, row in tqdm(users.iterrows(), total=users.shape[0], desc="Do NLP Preprocessing (stemming/Lemmatization/stopwords removal)"):
        text = nlp_preprocess_text(row['Body'])
        users.at[index, 'Body'] = text

    return users


'''
    prepare data - cut lower and upper limit of comments and export to csv
'''
def dataPreparation(users_df : pd.DataFrame, fixed_number_comments : int, plot=False, to_csv=False) -> pd.DataFrame:
    # remove none and empty entries
    users_df = users_df.replace(to_replace=['None', ''], value=np.nan).dropna()

    # if plot: showNrOfCommentsPerUser(users_df)
    # if plot: userStats(users_df, "All Users")

    # cut users with less than fixed_number_comments
    users_subset = CutUsersLowerLimit(users_df, fixed_number_comments)
    # if plot: userStats(users_subset, "Only relevant users")
    # if plot: showNrOfCommentsPerUserBarChart(users_subset)

    # cut users with more than fixed_number_comments
    users_subset = cutUsersUpperLimit(users_subset, fixed_number_comments)
    # if plot: userStats(users_subset, "All Users equal comment size")

    users_subset = calculateWritingTime(users_subset, plot=False)
    users_subset = doNLPpreprocessing(users_subset)

    if to_csv: users_subset.to_csv('dataset/prepared_corpus' + str(fixed_number_comments) + '.csv', index=False, sep=';')
    return users_subset

'''
    import csv corpus
'''
def getPreparedCorpus(fixed_number_comments : int) -> pd.DataFrame:
    try:
        print("Reading CSV data")
        users_df = pd.read_csv('dataset/prepared_corpus' + str(fixed_number_comments) + '.csv', sep=';')
        return users_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_PREPARED_CSV to False)", file=sys.stderr)
        sys.exit()
