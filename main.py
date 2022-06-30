import pandas as pd
import sqlite3
from vectorization import feature_matrix
from models import models
import preprocess.data_preprocessing
import argparse
from vectorization.feature_matrix import VectorizationType
from models.models import ModelType


# GOAL: try to identify specific posters on their writing style (or additional metadata)

# Step 1: get all Posts
# Step 2: preprocessing von allen posts
# Step 3: feature identification (corr matrix)
# Step 4: user identifizieren basierend auf writing style


def startConnection():
    print("Starting connection to DB")
    con = sqlite3.connect('dataset/corpus.sqlite3')
    articles_df = pd.read_sql_query("SELECT ID_Article, Path FROM Articles", con)

    users_df = pd.read_sql_query("SELECT ID_Post, ID_User, Body, ID_Article, CreatedAt, PositiveVotes, NegativeVotes FROM Posts", con)
    return users_df, articles_df


USE_PREPARED_CSV = False
USE_FEATURE_CSV = False
USE_METADATA = True
FIXED_NUMBER_COMMENTS = 1000
VECTORIZATIONTYPE = VectorizationType.Stylometry


def main():
    print("######################################### STEP 1 - IMPORT DATA ############################################")
    if USE_FEATURE_CSV:
        pass  # just for testing go directly to the model using a predefined feature matrix
    elif USE_PREPARED_CSV:
        users_df = preprocess.data_preprocessing.getPreparedCorpus(FIXED_NUMBER_COMMENTS)
    else:
        users_df, articles_df = startConnection()
        # preprocess the data - remove None and authors with < 50 comments and cut all authors to 50 comments
        users_df = preprocess.data_preprocessing.dataPreparation(users_df, articles_df, FIXED_NUMBER_COMMENTS, plot=False,
                                                                 to_csv=False)

    print("Import finished")
    print("########################## STEP 2 - CREATE WORD EMBEDDINGS / VECTORIZATION ################################")

    if USE_FEATURE_CSV:
        fm = feature_matrix.getFeatureMatrix()
    elif VECTORIZATIONTYPE == VectorizationType.Word2Vec:
        fm = feature_matrix.getModelInput(users_df, VECTORIZATIONTYPE,to_csv=False)
    else:
        fm = feature_matrix.getModelInput(users_df, VECTORIZATIONTYPE, to_csv=True)

    if VECTORIZATIONTYPE == VECTORIZATIONTYPE.Stylometry:
        # for metadata we use the time (in hours) of writing the comment, number of positive and negative votes
        fm = feature_matrix.addMetadataToMatrix(users_df, fm)

    print("######################################### STEP 3 - CREATE MODELS ##########################################")

    # if VECTORIZATIONTYPE == VectorizationType.NN:
    # models.createModelWithFeatureMatrix(fm, ModelType.NN, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.RANDOM, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.SVM, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.MLP,  vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.KNN, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.MNB, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.LR, vecType=VECTORIZATIONTYPE, print_report=True)


if __name__ == "__main__":
    main()
