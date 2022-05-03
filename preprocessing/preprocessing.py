
# first download stopwords
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from germalemma import GermaLemma
import pos_tagging
import pickle
import os

POS_TAGGING_GERMAN_PICKLE = 'nltk_german_classifier_data.pickle'

def removeStopwords(comment):
    stops = set(stopwords.words("german"))
    cleaned_comment = []
    for word in comment.split():
        if word not in stops:
            cleaned_comment.append(word)
    return " ".join(cleaned_comment)


def lemmatizeSentence(comment):
    lemmatizer = GermaLemma()
    tokens = nltk.word_tokenize(comment)

    if not os.path.exists(POS_TAGGING_GERMAN_PICKLE):
        tagger = pos_tagging.trainAndSave_POSModel()
    else:
        with open(POS_TAGGING_GERMAN_PICKLE, 'rb') as f:
            tagger = pickle.load(f)

    tokens_with_pos = pos_tagging.POSTaggingWithTagger(tagger, tokens)
    return ' '.join([lemmatizer.find_lemma(element[0], element[1]) for element in tokens_with_pos])


def main():

    lem_tok = lemmatizeSentence("Hallo, ich war mal größer")
    print(lem_tok)
    pass

if __name__ == "__main__":
        main()

