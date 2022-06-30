import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from vectorization import vectorization, stylometry
import time
from tqdm import tqdm
from enum import Enum
import utils
import sys
from sklearn.preprocessing import OneHotEncoder


class VectorizationType(Enum):
    Stylometry = 1
    BagOfWords = 2
    TfIdf = 4



def addMetadataToMatrix(users_df: pd.DataFrame, fm: pd.DataFrame):
    """
        take the metadata of a comment from users_df and add it to the feature_matrix
    """
    fm['PositiveVotes'] = users_df['PositiveVotes']
    fm['NegativeVotes'] = users_df['NegativeVotes']

    # one hot encoding for writing time
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(users_df[['WritingTime']]).toarray())
    encoder_df.columns = ['WritingTime.Morning', 'WritingTime.Midday', 'WritingTime.Afternoon', 'WritingTime.Evening',
                          'WritingTime.Night']
    fm = fm.join(encoder_df)

    # topics one-hot-encoded
    topics_df = pd.DataFrame(encoder.fit_transform(users_df[['Topic']]).toarray())
    column_labels = [ "topic"+ str(i) for i in range(topics_df.shape[1])]
    topics_df.columns = column_labels
    fm = fm.join(topics_df)

    return fm




def createFeatureMatrix(users_df: pd.DataFrame, to_csv=False):
    """
        main function to create the feature matrix for stylometry
        @return: a dataframe with all users as rows and all features as columns
    """

    start = time.time()

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


    print("Found in time [s] the feature matrix: " + str(time.time() - start))
    if to_csv: saveFeatureMatrix(feature_matrix)
    return feature_matrix


def saveFeatureMatrix(feature_matrix : pd.DataFrame):
    feature_matrix.to_csv('dataset/feature_matrix.csv', index=False, sep=';')


def normalizeFeatureMatrix(featureMatrix : pd.DataFrame) -> pd.DataFrame:
    user_id_col = featureMatrix["ID_User"]
    normalized_matrix = (featureMatrix-featureMatrix.min())/(featureMatrix.max()-featureMatrix.min())
    normalized_matrix['ID_User'] = user_id_col
    return normalized_matrix




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
    """
        import the feature matrix from earlier
    """
    try:
        print("Reading feature matrix")
        feature_df = pd.read_csv('dataset/feature_matrix.csv', sep=';')
        return feature_df
    except FileNotFoundError:
        print("[ERROR] You first need to create the CSV file (set USE_FEATURE_CSV to False)", file=sys.stderr)
        sys.exit()
