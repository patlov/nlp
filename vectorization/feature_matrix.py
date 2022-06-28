import pandas as pd
from sklearn.model_selection import train_test_split
from vectorization import vectorization, stylometry
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    # feature_matrix = pd.DataFrame()

    start = time.time()

    # BagOfWords feature extraction
    if type == VectorizationType.BagOfWords or type == VectorizationType.TfIdf:
        cleaned_text = []
        for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0],
                               desc=f'Preprocessing and generating Training and Testing Set'):
            try:
                text = row['Body']
                if nlp_preprocess:
                    text = nlp_preprocess_text(text)  # add features to dict
                cleaned_text.append(text)
            except Exception as e:
                utils.writeToErrorLog("Error at TfIdf for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        users_df['cleaned'] = cleaned_text

        X = users_df['cleaned'].values
        y = users_df['ID_User'].values

        train_comments, test_comments, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if type == VectorizationType.BagOfWords:
            vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        else:
            vectorizer = CountVectorizer(ngram_range=(2, 2))
        vectorizer.fit(train_comments)
        train_comments_vec = vectorizer.transform(train_comments)
        test_comments_vec = vectorizer.transform(test_comments)

        print("Found in time [s] the feature matrix: " + str(time.time() - start))
        return [train_comments_vec, test_comments_vec, y_train, y_test]
    # Stylometry feature extraction
    elif type == VectorizationType.Stylometry:
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
        col = 'ID_User'
        y = feature_matrix[col]
        X = feature_matrix.loc[:, feature_matrix.columns != col]

        classes = feature_matrix.ID_User.unique()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return [X_train, X_test, y_train, y_test]
    # if to_csv: feature_matrix.to_csv('dataset/feature_matrix.csv', index=False, sep=';')
    else:
        assert "You should pick a valid Vectorizationtype"

    # return feature_matrix


def getFeatureMatrix() -> pd.DataFrame:
    try:
        print("Reading feature matrix")
        feature_df = pd.read_csv('dataset/feature_matrix.csv', sep=';')
        return feature_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_FEATURE_CSV to False)", file=sys.stderr)
        sys.exit()


def preprocessDataFrame(users_df: pd.DataFrame, to_csv: bool = False) -> pd.DataFrame:
    cleaned_text = []
    for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0], desc="Preprocessing Dataframe"):
        try:
            text = row['Body']
            text = nlp_preprocess_text(text)  # add features to dict
            cleaned_text.append(text)
        except Exception as e:
            utils.writeToErrorLog("Error at Preprocessing Dataframe " + str(row['ID_Post']) + "::" + str(e) + "\n")
    users_df['cleaned'] = cleaned_text
    if to_csv: users_df.to_csv('dataset/users_df.csv', index=False, sep=';')
    return users_df


def getSavedDF() -> pd.DataFrame:
    try:
        print("Reading users dataframe from local drive")
        feature_df = pd.read_csv('dataset/users_df.csv', sep=';')
        return feature_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_EXISTING_DF to False)", file=sys.stderr)
        sys.exit()
