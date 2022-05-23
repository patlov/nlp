import nltk
import random
import pickle
import ClassifierBasedGermanTagger

'''
    once the model must be trained and saved 
'''


def trainPOSModel():
    corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')
    tagged_sents = list(corp.tagged_sents())
    random.shuffle(tagged_sents)

    # set a split size: use 90% for training, 10% for testing
    split_perc = 0.1
    split_size = int(len(tagged_sents) * split_perc)
    train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

    tagger = ClassifierBasedGermanTagger.ClassifierBasedGermanTagger(train=train_sents)

    accuracy = tagger.accuracy(test_sents)
    print("Trained the model with accuracy:", accuracy)

    return tagger


'''
    POS Tagging using the preprocessed classifier
'''


def POSTagging(text_tokens: list):
    with open('dataset/nltk_german_classifier_data.pickle', 'rb') as f:
        tagger = pickle.load(f)

    text_tags = tagger.tag(text_tokens)
    return text_tags


'''
    POS Tagging using a delivered tagger via parameters
'''


def POSTaggingWithTagger(tagger, word: list):
    return tagger.tag(word)


'''
    just test the functions
'''


def main():
    # train and save tagger
    # tagger = trainPOSModel()
    # with open('nltk_german_classifier_data.pickle', 'wb') as f:
    #     pickle.dump(tagger, f)

    # load tagger
    with open('dataset/nltk_german_classifier_data.pickle', 'rb') as f:
        tagger = pickle.load(f)

    tags = POSTaggingWithTagger(tagger, "ich war mal größer".split(sep=' '))
    print(tags)
    pass


if __name__ == "__main__":
    main()
