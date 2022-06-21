import pandas as pd
import re


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
