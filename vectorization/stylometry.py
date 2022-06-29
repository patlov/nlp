import pandas as pd
from lexicalrichness import LexicalRichness

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


def getLexicalRichness(text: str):
    try:
        lex = LexicalRichness(text)
        herdan = lex.Herdan
        summer = lex.Summer
        maas = lex.Maas
        mtld = lex.mtld()
    except:
        herdan = 0
        summer = 0
        maas = 0
        mtld = 0
    return herdan, summer, maas, mtld


def getSentencesLength(text: str) -> int:
    return len(text.split())


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
    herdan_diversity, summer_diversity, maas_diversity, lexical_richness = getLexicalRichness(text)
    word_count = getSentencesLength(text)

    features = {
        "letter_ratio": letters_ratio,
        "digit_ratio": digit_ration,
        "uppercase_ratio": uppercase_ration,
        "lowercase_ratio": lowercase_ration,
        "whitespace_ratio": whitespace_ration,
        "herdan_diversity": herdan_diversity,
        "summer_diversity": summer_diversity,
        "maas_diversity": maas_diversity,
        "lexical_richness": lexical_richness,
        "word_count": word_count
    }

    return features
