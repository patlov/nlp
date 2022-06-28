import pandas as pd

from vectorization import vectorization, stylometry
import time
from tqdm import tqdm
from enum import Enum
import utils
import sys
from preprocess.nlp_preprocessing import nlp_preprocess_text


class VectorizationType(Enum):
    Stylometry = 1
    BagOfWords = 2
    Word2Vec = 3
    TfIdf = 4


'''
    main function to create different types of feature matrix's
    @return: a dataframe with all users as rows and all features as columns
'''


def getModelInput(users_df: pd.DataFrame, type: VectorizationType, nlp_preprocess=False,
                  to_csv=False):
    feature_matrix = pd.DataFrame()

    start = time.time()

    # Stylometry feature extraction
    if type == VectorizationType.Stylometry:
        tmp_feature_list = []
        for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0], desc="Calculating Stylometry"):
            try:
                text = row['Body']
                if nlp_preprocess:
                    text = nlp_preprocess_text(text)
                features = {'ID_User': row['ID_User']}
                features.update(stylometry.createStylometryFeatures(text))
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at stylometry for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list)

    # BagOfWords feature extraction
    elif type == VectorizationType.BagOfWords:
        tmp_feature_list = []
        for index, row in tqdm(users_df[:100].iterrows(), total=users_df.shape[0], desc="Calculating BagOfWords"):
            try:
                text = row['Body']
                if nlp_preprocess:
                    text = nlp_preprocess_text(text)
                features = {'ID_User': row['ID_User']}
                features.update(vectorization.bagOfWords(text))  # add features to dict
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at BagOfWords for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list).fillna(0)  # fill NaN with 0

    # TfIdf feature extraction
    elif type == VectorizationType.TfIdf:
        tmp_feature_list = []
        for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0], desc="Calculating TfIdf"):
            try:
                text = row['Body']
                if nlp_preprocess:
                    text = nlp_preprocess_text(text)
                features = {'ID_User': row['ID_User']}
                feat = vectorization.TfIdf(text)
                features.update(vectorization.TfIdf(text))  # add features to dict
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at TfIdf for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list).fillna(0)  # fill NaN with 0

    # Word2Vec list of list
    elif type == VectorizationType.Word2Vec:
        cleaned_text = []
        for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0], desc="Calculating TfIdf"):
            try:
                text = row['Body']
                if nlp_preprocess:
                    text = nlp_preprocess_text(text)  # add features to dict
                cleaned_text.append(text)
            except Exception as e:
                utils.writeToErrorLog("Error at TfIdf for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        users_df['cleaned'] = cleaned_text

    print("Found in time [s] the feature matrix: " + str(time.time() - start))
    if to_csv: feature_matrix.to_csv('dataset/feature_matrix.csv', index=False, sep=';')
    return feature_matrix


def getFeatureMatrix() -> pd.DataFrame:
    try:
        print("Reading feature matrix")
        feature_df = pd.read_csv('dataset/feature_matrix.csv', sep=';')
        return feature_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_FEATURE_CSV to False)", file=sys.stderr)
        sys.exit()
