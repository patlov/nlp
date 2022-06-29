import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from vectorization import vectorization, stylometry
import time
from tqdm import tqdm
from enum import Enum
import utils
import sys


class VectorizationType(Enum):
    Stylometry = 1
    BagOfWords = 2
    Word2Vec = 3
    TfIdf = 4
    NN = 5


def addMetadataToMatrix(users_df: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    fm['PositiveVotes'] = users_df['PositiveVotes']
    fm['NegativeVotes'] = users_df['NegativeVotes']
    fm['WritingTime'] = users_df['WritingTime']
    return fm


'''
    main function to create different types of feature matrix's
    @return: a dataframe with all users as rows and all features as columns
'''


def getModelInput(users_df: pd.DataFrame, type: VectorizationType, to_csv=False):
    feature_matrix = pd.DataFrame()

    start = time.time()

    # Stylometry feature extraction
    if type == VectorizationType.Stylometry:
        tmp_feature_list = []
        for index, row in tqdm(users_df.iterrows(), total=users_df.shape[0], desc="Calculating Stylometry"):
            try:
                text = row['Body']
                features = {'ID_User': row['ID_User']}
                features.update(stylometry.createStylometryFeatures(text))
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at stylometry for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list)

    # NN we need only the users_df
    elif type == VectorizationType.NN or type == VectorizationType.TfIdf or type == VectorizationType.BagOfWords:
        feature_matrix = users_df

    print("Found in time [s] the feature matrix: " + str(time.time() - start))
    if to_csv: feature_matrix.to_csv('dataset/feature_matrix.csv', index=False, sep=';')
    return feature_matrix


def covertTextToNumeric(x_train, x_test, features=30000):
    tokenizer = Tokenizer(lower=True, split=" ", num_words=features)
    tokenizer.fit_on_texts(x_train)
    x_train_vec = tokenizer.texts_to_sequences(x_train)
    x_test_vec = tokenizer.texts_to_sequences(x_test)
    max_length = max([len(x) for x in x_train_vec])
    x_train_vec = sequence.pad_sequences(x_train_vec, maxlen=max_length, padding="post")
    x_test_vec = sequence.pad_sequences(x_test_vec, maxlen=max_length, padding="post")

    return [x_train_vec, x_test_vec, max_length]


def getFeatureMatrix() -> pd.DataFrame:
    try:
        print("Reading feature matrix")
        feature_df = pd.read_csv('dataset/feature_matrix.csv', sep=';')
        return feature_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_FEATURE_CSV to False)", file=sys.stderr)
        sys.exit()
