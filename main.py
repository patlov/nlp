import pandas as pd
import sqlite3
from vectorization import feature_matrix
import preprocess.data_preprocessing
import argparse
from vectorization.feature_matrix import VectorizationType


# GOAL: try to identify specific posters on their writing style (or additional metadata)

# Step 1: get all Posts
# Step 2: preprocessing von allen posts
# Step 3: feature identification (corr matrix)
# Step 4: user identifizieren basierend auf writing style

def mergeDF(articles, posts):
    return pd.merge(articles, posts, on='ID_Article')




def startConnection():
    print("Starting connection to DB")
    con = sqlite3.connect('dataset/corpus.sqlite3')
    # articles_df = pd.read_sql_query("SELECT * FROM Articles", con)
    # posts_df = pd.read_sql_query("SELECT * FROM Posts", con)

    users_df = pd.read_sql_query("SELECT ID_Post, ID_User, Body FROM Posts ORDER BY ID_User", con)
    return users_df

USE_CSV = True

def main():
    # parser = argparse.ArgumentParser(description='NLP SS 2022 - 1 Million Post Dataset from derStandard')
    # parser.add_argument('--csv',  required=False, type=bool, help='use prepared csv dataset')
    # parser.add_argument('--vec', required=True, help='type of vectorization to create the feature matrix')
    # args = parser.parse_args()

    print("######################################### STEP 1 - IMPORT DATA ############################################")
    if USE_CSV:
        users_df = preprocess.data_preprocessing.getPreparedCorpus()
    else:
        users_df = startConnection()
        # preprocess the data - remove None and authors with < 50 comments and cut all authors to 50 comments
        users_df = preprocess.data_preprocessing.dataPreparation(users_df, plot=False, to_csv=True)

    print("Import finished")
    print("########################## STEP 2 - CREATE WORD EMBEDDINGS / VECTORIZATION ################################")


    fm = feature_matrix.createFeatureMatrix(users_df, VectorizationType.BagOfWords)



    print("######################################### STEP 3 - CREATE MODELS ##########################################")

    '''
        uncomment this to see performance of our system with LinearSVC model and top 100 authors and their 500 comments
    '''
    #models.getTopAuthorComments(users_df, 80, 500)


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
