import pandas as pd
from vectorization import text_properties
from vectorization import vectorization
import time
from tqdm import tqdm
from enum import Enum
import utils

class VectorizationType(Enum):
    Stylometry = 1
    BagOfWords = 2
    Word2Vec = 3
    TfIdf = 4



'''
    extract all features from the user's comments
    @return: one user with all features calculated
'''
def createStylometryFeaturesPerUser(df_user: pd.DataFrame) -> list:
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
    extract stylometry features from text
    @return: the calculated features from one comment
'''
def createStylometryFeatures(text: str) -> dict:


    letters_ratio = text_properties.getLettersRatio(text)
    digit_ration = text_properties.getDigitRatio(text)
    uppercase_ration = text_properties.getUppercaseRatio(text)
    lowercase_ration = text_properties.getLowercaseRatio(text)
    whitespace_ration = text_properties.getWhitespaceRatio(text)

    features = {
        "letter_ratio": letters_ratio,
        "digit_ration": digit_ration,
        "uppercase_ration": uppercase_ration,
        "lowercase_ration": lowercase_ration,
        "whitespace_ration": whitespace_ration
    }

    return features


'''
    main function to create different types of feature matrix's
    @return: a dataframe with all users as rows and all features as columns
'''
def createFeatureMatrix(users_df: pd.DataFrame, type : VectorizationType) -> pd.DataFrame:
    feature_matrix = pd.DataFrame()

    start = time.time()

    # Stylometry feature extraction
    if type == VectorizationType.Stylometry:
        tmp_feature_list = []
        for index, row in tqdm(users_df[:1000].iterrows(), total=users_df.shape[0], desc="Calculating Stylometry"):
            try:
                features = createStylometryFeatures(row['Body'])
                features['ID_Post'] = row['ID_Post']
                features['ID_User'] = row['ID_User']
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at stylometry for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list)



    # BagOfWords feature extraction
    elif type == VectorizationType.BagOfWords:
        tmp_feature_list = []
        for index, row in tqdm(users_df[:1000].iterrows(), total=users_df.shape[0], desc="Calculating BagOfWords"):
            try:
                features = vectorization.bagOfWords(row['Body']) # add features to dict
                features['ID_Post'] = row['ID_Post']
                features['ID_User'] = row['ID_User']
                tmp_feature_list.append(features)
            except Exception as e:
                utils.writeToErrorLog("Error at BagOfWords for comment " + str(row['ID_Post']) + "::" + str(e) + "\n")
        feature_matrix = pd.DataFrame(tmp_feature_list).fillna(0) # fill NaN with 0




    # Word2Vec feature extraction
    elif type == VectorizationType.Word2Vec:
        pass

    # TfIdf feature extraction
    elif type == VectorizationType.TfIdf:
        pass



    print("Found in time [s] the feature matrix: " + str(time.time() - start))
    return feature_matrix

