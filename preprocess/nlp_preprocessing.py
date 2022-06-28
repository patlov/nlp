# first download stopwords
import pickle
import string

import nltk
from germalemma import GermaLemma
from nltk.corpus import stopwords
from HanTa import HanoverTagger as ht
# nltk.download('stopwords')
from preprocess import pos_tagging
import spacy
nlp = spacy.load('de_core_news_md')

tagger = ht.HanoverTagger('morphmodel_ger.pgz')

# POS_TAGGING_GERMAN_PICKLE = 'dataset/nltk_german_classifier_data.pickle'
# with open('dataset/nltk_german_classifier_data.pickle', 'rb') as f:
#    tagger = pickle.load(f)


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
    # approach using hanta tagger
    # lemma = [lemma for (word, lemma, pos) in tagger.tag_sent(comment.split())]
    # return ' '.join(lemma)

    # approach using spacy
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
