import pandas as pd
import numpy as np
import sqlite3
import text_properties
import models
import time
import preprocess.data_exploration


# GOAL: try to identify specific posters on their writing style (or additional metadata)

# Step 1: get all Posts
# Step 2: preprocessing von allen posts
# Step 3: feature identification (corr matrix)
# Step 4: user identifizieren basierend auf writing style

def mergeDF(articles, posts):
    return pd.merge(articles, posts, on='ID_Article')


'''
    extract all features from the user's comments
    @return: one user with all features calculated
'''


def featureExtraction(df_user: pd.DataFrame) -> list:
    # user_df is a dataframe with all comments from one user

    # make feature extraction
    average_text_length = text_properties.getAverageTextLength(df_user)

    # go through all comments of a user, calculate the features and return it as dict
    current_user_features = []
    for index, row in df_user.iterrows():
        text = row['Body']

        letters_ratio = text_properties.getLettersRatio(text)
        digit_ration = text_properties.getDigitRatio(text)
        uppercase_ration = text_properties.getUppercaseRatio(text)
        lowercase_ration = text_properties.getLowercaseRatio(text)
        whitespace_ration = text_properties.getWhitespaceRatio(text)

        features = {
            "ID_Post": row['ID_Post'],
            "ID_User": row['ID_User'],
            "letter_ratio": letters_ratio,
            "digit_ration": digit_ration,
            "uppercase_ration": uppercase_ration,
            "lowercase_ration": lowercase_ration,
            "whitespace_ration": whitespace_ration
        }
        current_user_features.append(features)

    # return list of feature values for this user
    return current_user_features


'''
    create the features for all users
    @return: a dataframe with all users as rows and all features as columns
'''


def createFeatureMatrix(all_users_df: pd.DataFrame) -> pd.DataFrame:
    feature_matrix = pd.DataFrame()
    user_ids = all_users_df.ID_User.unique()
    start = time.time()
    for user_id in user_ids[:1000]:
        user_subset = all_users_df.loc[all_users_df['ID_User'] == user_id]
        current_user_comments_with_features = featureExtraction(user_subset)

        user_comments_feature_matrix = pd.DataFrame(current_user_comments_with_features)
        feature_matrix = feature_matrix.append(user_comments_feature_matrix)
    print("Found in time [s] the feature matrix: " + str(time.time() - start))
    return feature_matrix

    print(keep_comments)


def startConnection():
    con = sqlite3.connect('dataset/corpus.sqlite3')
    # articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    # posts_df = pd.read_sql_query("SELECT * FROM Posts", con)

    users_df = pd.read_sql_query("SELECT ID_Post, ID_User, Body FROM Posts ORDER BY ID_User", con)
    return users_df

def main():

    users_df = startConnection()

    users_df = preprocess.data_exploration.preprocessingSteps(users_df, plot=False) # preprocess the data - remove None and authors with < 50 comments

    '''
        uncomment this to see performance of our system with LinearSVC model and top 100 authors and their 500 comments
    '''
    models.getTopAuthorComments(users_df, 80, 500)

    '''
        uncomment this to see performance of our system with only preprocessing the texts and using the MNB model, takes some time
    '''
    # print('-' * 42)
    # print('Results for Model without Feature Matrix, but using Countvectorization: ')
    # print('-' * 42)
    # use only first 10000 entries because we cannot get enough memory on the machine...
    # models.createModelWihoutFeatureMatrix(users_df.head(1000))


    '''
        uncomment this to see performance of our SVM, MNB models with our own feature matrix
    '''
    # features_matrix = createFeatureMatrix(users_df)
    # print('-' * 42)
    # print('Results for Model with Support Vector Machines are: ')
    # models.createModelWithFeatureMatrix(features_matrix, 'SVM')
    # print('-' * 42)
    # print('Results for Model with Multinomial Naive Bayes are: ')
    # models.createModelWithFeatureMatrix(features_matrix, 'MNB')
    # print('-' * 42)
    # print('Results for Model with Linear Regression are: ')
    # models.createModelWithFeatureMatrix(features_matrix, 'LR')


if __name__ == "__main__":
    main()
