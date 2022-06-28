import pandas as pd
import sqlite3
from vectorization import feature_matrix
from models import models
import preprocess.data_preprocessing
import argparse
from vectorization.feature_matrix import VectorizationType
from models.models import ModelType
from preprocess.nlp_preprocessing import nlp_preprocess_text


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


USE_PREPARED_CSV = False
USE_FEATURE_CSV = False
USE_EXISTING_DF = False
FIXED_NUMBER_COMMENTS = 100
VECTORIZATIONTYPE = VectorizationType.TfIdf


def main():
    # todo possibly add arguments instead of global vars
    # parser = argparse.ArgumentParser(description='NLP SS 2022 - 1 Million Post Dataset from derStandard')
    # parser.add_argument('--csv',  required=False, type=bool, help='use prepared csv dataset')
    # parser.add_argument('--vec', required=True, help='type of vectorization to create the feature matrix')
    # args = parser.parse_args()

    print("######################################### STEP 1 - IMPORT DATA ############################################")
    if USE_FEATURE_CSV:
        pass  # just for testing go directly to the model using a predefined feature matrix
    elif USE_PREPARED_CSV:
        users_df = preprocess.data_preprocessing.getPreparedCorpus(FIXED_NUMBER_COMMENTS)
    else:
        users_df = startConnection()
        # preprocess the data - remove None and authors with < 50 comments and cut all authors to 50 comments
        users_df = preprocess.data_preprocessing.dataPreparation(users_df, FIXED_NUMBER_COMMENTS, plot=False,
                                                                 to_csv=False)

    print("Import finished")
    print("########################## STEP 2 - CREATE WORD EMBEDDINGS / VECTORIZATION ################################")

    if USE_FEATURE_CSV:
        fm = feature_matrix.getFeatureMatrix()
    else:
        [X_train, X_test, y_train, y_test] = feature_matrix.getModelInput(users_df, VECTORIZATIONTYPE,
                                                                          nlp_preprocess=True, to_csv=False)

    print("######################################### STEP 3 - CREATE MODELS ##########################################")

    if VECTORIZATIONTYPE != VectorizationType.Word2Vec:
        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.RANDOM, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.SVM, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.MNB, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.LR, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.MLP, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.XG, print_report=True)

        models.createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, ModelType.KNN, print_report=True)
    else:
        models.createW2VDeepLearningModel(users_df)


if __name__ == "__main__":
    main()
