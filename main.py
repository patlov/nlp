import pandas as pd
import sqlite3
from vectorization import feature_matrix
from models import models
import preprocess.data_preprocessing
from vectorization.feature_matrix import VectorizationType
from models.models import ModelType


def startConnection():
    print("Starting connection to DB")
    con = sqlite3.connect('dataset/corpus.sqlite3')
    articles_df = pd.read_sql_query("SELECT ID_Article, Path FROM Articles", con)

    users_df = pd.read_sql_query("SELECT ID_Post, ID_User, Body, ID_Article, CreatedAt, PositiveVotes, NegativeVotes FROM Posts", con)
    return users_df, articles_df


USE_PREPARED_CSV = False # for debugging, use a CSV version of the users_df created earlier
USE_FEATUREMATRIX_CSV = False # for debugging, use a CSV version of the feature_matrix created earlier
FIXED_NUMBER_COMMENTS = 1000

""" Change here the type of vectorization you want to aplly """
VECTORIZATIONTYPE = VectorizationType.Stylometry


def main():
    print("######################################### STEP 1 - IMPORT DATA ############################################")

    if USE_FEATUREMATRIX_CSV:
        pass
    elif USE_PREPARED_CSV:
        users_df = preprocess.data_preprocessing.getPreparedCorpus(FIXED_NUMBER_COMMENTS)
    else:
        users_df, articles_df = startConnection()
        # preprocess the data - remove None and authors with < 50 comments and cut all authors to 50 comments
        users_df = preprocess.data_preprocessing.dataPreparation(users_df, articles_df, FIXED_NUMBER_COMMENTS,
                                                                 plot=False, to_csv=False)

    print("Import finished")
    print("########################## STEP 2 - CREATE WORD EMBEDDINGS / VECTORIZATION ################################")

    if VECTORIZATIONTYPE == VECTORIZATIONTYPE.Stylometry:
        # for other vectorizations the feature matrix is calculated directly at the model creation

        if USE_FEATUREMATRIX_CSV:
            users_df = preprocess.data_preprocessing.getPreparedCorpus(FIXED_NUMBER_COMMENTS)
            fm = feature_matrix.getFeatureMatrix()
        else:
            fm = feature_matrix.createFeatureMatrix(users_df, to_csv=True)

        # for metadata we use the time (in hours) of writing the comment, number of positive and negative votes
        fm = feature_matrix.addMetadataToMatrix(users_df, fm)
        fm = feature_matrix.normalizeFeatureMatrix(fm)
        #feature_matrix.saveFeatureMatrix(fm)
    else:
        users_df = preprocess.data_preprocessing.getPreparedCorpus(FIXED_NUMBER_COMMENTS)
        fm = users_df

    print("######################################### STEP 3 - CREATE MODELS ##########################################")

    # if VECTORIZATIONTYPE == VectorizationType.Stylometry:
        # models.createModelWithFeatureMatrix(fm, ModelType.NN, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.RANDOM, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.SVM, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.MLP,  vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.KNN, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.MNB, vecType=VECTORIZATIONTYPE, print_report=True)

    models.createModelWithFeatureMatrix(fm, ModelType.LR, vecType=VECTORIZATIONTYPE, print_report=True)


if __name__ == "__main__":
    main()
