# first download stopwords
import pickle
import string
import re
import nltk
from germalemma import GermaLemma
from nltk.corpus import stopwords

# nltk.download('stopwords')
from preprocess import pos_tagging
import spacy
nlp = spacy.load('de_core_news_md')


def nlp_preprocess_text(text: str, rmStopwords: bool = True, rmPunctation: bool = True,
                        lemmatizeText: bool = True) -> str:
    if rmPunctation:
        text = removePunctation(text)
    if rmStopwords:
        text = removeStopwords(text)
    if lemmatizeText:
        text = lemmatizeSentence(text)

    return text


'''
    Remove stopwords like 'aber', 'und', ... 
'''


def removeStopwords(comment: str):
    stops = set(stopwords.words("german"))
    return " ".join([word for word in comment.split() if word not in stops])
    # cleaned_comment = []
    # for word in comment.split():
    #    if word not in stops:
    #        cleaned_comment.append(word)
    # return " ".join(cleaned_comment)


'''
    Remove sentence endings like '.' or '!'
'''


def removePunctation(text: str):
    nopunct = [char for char in text if char not in string.punctuation]
    return ''.join(nopunct)


'''
    Split a sentence into words
'''


def tokenize(text: str):
    return nltk.word_tokenize(text)


'''
    Lemmatization of words - convert e.g. "Ich war mal größer" to -> "Ich sein mal groß"
'''


def lemmatizeSentence(comment: str):
    return ' '.join([token.lemma_ for token in nlp(comment)])


'''
    Just testing the functions
'''


def main():
    # lem_tok = lemmatizeSentence("Hallo, ich war mal größer")
    # print(lem_tok)
    text = removeStopwords("Hallo, ich war mal groß und jetzt bin ich klein. Na und. Was machst du so?")
    print(text)
    pass


if __name__ == "__main__":
    main()
