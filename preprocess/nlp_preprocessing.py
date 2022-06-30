
import string
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

import spacy
nlp = spacy.load('de_core_news_md')



def nlp_preprocess_text(text: str, rmStopwords: bool = True, rmPunctation: bool = True,
                        lemmatizeText: bool = True) -> str:
    """
        main function to do all nlp preprocessing steps to a given text
    """
    if rmPunctation:
        text = removePunctation(text)
    if rmStopwords:
        text = removeStopwords(text)
    if lemmatizeText:
        text = lemmatizeSentence(text)

    return text



def removeStopwords(comment: str):
    """
        Remove stopwords like 'aber', 'und', ...
    """
    stops = set(stopwords.words("german"))
    return " ".join([word for word in comment.split() if word not in stops])


def removePunctation(text: str):
    """
        Remove sentence endings like '.' or '!'
    """
    nopunct = [char for char in text if char not in string.punctuation]
    return ''.join(nopunct)


def tokenize(text: str):
    """
        Split a sentence into words
    """
    return nltk.word_tokenize(text)



def lemmatizeSentence(comment: str):
    """
        Lemmatization of words - convert e.g. "Ich war mal größer" to -> "Ich sein mal groß"
    """
    return ' '.join([token.lemma_ for token in nlp(comment)])


