import pandas as pd



'''
# AVERAGE TEXT CALCULATIONS - properties for a list of texts
'''
def getAverageTextLength(df: pd.DataFrame):
    return df['Body'].str.len().mean()


# fraction of letters in the texts
def getAverageLettersRatio(df: pd.DataFrame):
    ratio_per_comment = df['Body'].str.count(r'[A-Za-z]') / df['Body'].str.len()
    return ratio_per_comment.mean()


# fraction of digits in the texts
def getAverageDigitsRatio(df: pd.DataFrame):
    ratio_per_comment = df['Body'].str.count(r'[1-9]') / df['Body'].str.len()
    return ratio_per_comment.mean()


# fraction of uppercase chars in the texts
def getAverageUppercaseRatio(df: pd.DataFrame):
    ratio_per_comment = df['Body'].str.count(r'[A-Z]') / df['Body'].str.len()
    return ratio_per_comment.mean()


# fraction of uppercase chars in the texts
def getAverageWhitespaceRatio(df: pd.DataFrame):
    ratio_per_comment = df['Body'].str.count(r'[ ]') / df['Body'].str.len()
    return ratio_per_comment.mean()


'''
# SINGLE TEXT CALCULATIONS
'''
def getLettersRatio(text: str) -> float:
    return sum(1 for c in text if c.isalpha()) / len(text)


def getDigitRatio(text: str) -> float:
    return sum(1 for c in text if c.isdigit()) / len(text)


def getUppercaseRatio(text: str) -> float:
    return sum(1 for c in text if c.isupper()) / len(text)


def getLowercaseRatio(text: str) -> float:
    return sum(1 for c in text if c.islower()) / len(text)


def getWhitespaceRatio(text: str) -> float:
    return sum(1 for c in text if c.isspace()) / len(text)


'''
    extract all features from the user's comments
    @return: one user with all features calculated
'''
def createStylometryFeaturesPerUser(df_user: pd.DataFrame) -> list:
    # user_df is a dataframe with all comments from one user

    # make feature extraction
    average_text_length = getAverageTextLength(df_user)

    # go through all comments of a user, calculate the features and return it as dict
    current_user_features = []
    for index, row in df_user.iterrows():
        text = row['Body']

        letters_ratio = getLettersRatio(text)
        digit_ration = getDigitRatio(text)
        uppercase_ration = getUppercaseRatio(text)
        lowercase_ration = getLowercaseRatio(text)
        whitespace_ration = getWhitespaceRatio(text)

        # todo add here further stylometry features, specially vocabulary richness measures

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


    letters_ratio = getLettersRatio(text)
    digit_ration = getDigitRatio(text)
    uppercase_ration = getUppercaseRatio(text)
    lowercase_ration = getLowercaseRatio(text)
    whitespace_ration = getWhitespaceRatio(text)

    features = {
        "letter_ratio": letters_ratio,
        "digit_ration": digit_ration,
        "uppercase_ration": uppercase_ration,
        "lowercase_ration": lowercase_ration,
        "whitespace_ration": whitespace_ration
    }

    return features
