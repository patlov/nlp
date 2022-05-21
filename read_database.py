import pandas as pd
import sqlite3
import text_properties
from User import User


# GOAL: try to identify specific posters on their writing style (or additional metadata)

# Step 1: get all Posts from a specific user-id
# Step 2: preprocessing von allen posts von einem user
# Step 3: feature identification (corr matrix)
# Step 4: user identifizieren basierend auf writing style

def mergeDF(articles, posts):
    return pd.merge(articles, posts, on='ID_Article')


'''
    extract all features from the user's comments
    @return: one user with all features calculated
'''
def featureExtraction(df_user : pd.DataFrame) -> list:
    # user_df is a dataframe with all comments from one user

    # make feature extraction
    average_text_length = text_properties.getAverageTextLength(df_user)

    # go through all comments of a user, calculate the features and return it as dict
    current_user_features = []
    for index, row in df_user.iterrows():
        text = row['Body']
        if text == None:
            continue
        letters_ratio = text_properties.getLettersRatio(text)
        digit_ration = text_properties.getDigitRatio(text)
        uppercase_ration = text_properties.getUppercaseRatio(text)
        lowercase_ration = text_properties.getLowercaseRatio(text)
        whitespace_ration = text_properties.getWhitespaceRatio(text)

        features = {
            "ID_Post": row['ID_Post'],
            "ID_User" : row['ID_User'],
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
def createFeatureMatrix(all_users_df : pd.DataFrame) -> pd.DataFrame:

    feature_matrix = pd.DataFrame()
    user_ids = all_users_df.ID_User.unique()
    for user_id in user_ids[:10]:
        user_subset = all_users_df.loc[all_users_df['ID_User'] == user_id]
        current_user_comments_with_features = featureExtraction(user_subset)

        user_comments_feature_matrix = pd.DataFrame(current_user_comments_with_features)
        feature_matrix = feature_matrix.append(user_comments_feature_matrix)

    return feature_matrix


def main():
    con = sqlite3.connect('dataset/corpus.sqlite3')

    # articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    # posts_df = pd.read_sql_query("SELECT * FROM Posts", con)

    users_df = pd.read_sql_query("SELECT ID_Post, ID_User, Body FROM Posts ORDER BY ID_User", con)
    features_matrix = createFeatureMatrix(users_df)



    # maybe useful for metadata
    # newspaper_staff_df = pd.read_sql_query("SELECT * FROM Newspaper_Staff", con)
    # annotations_df = pd.read_sql_query("SELECT * FROM Annotations", con)
    # annotations_consolidated_df = pd.read_sql_query("SELECT * FROM Annotations_consolidated", con)
    # cross_val_split_df = pd.read_sql_query("SELECT * FROM CrossValSplit", con)
    # categories_df = pd.read_sql_query("SELECT * FROM Categories", con)

    # print(users)

    # full_df = mergeDF(articles_df, posts_df)
    # print(full_df)


if __name__ == "__main__":
    main()
