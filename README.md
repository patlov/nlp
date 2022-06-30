# Natural Language Processing SS 2022
Author Attribution of "DerStandard Forum Writing Style"

``` Download stopwords first in preprocessing```


##Setup
First you need to install all the packes from the [requirements.txt](requirements.txt) file.
Maybe you create first a virtual environment and install then in there or install locally the packages.
The easiest way to install the required packages would be:

```angular2html
pip install -r requirements.txt
```

and for the lemmatizer for the german language you need to download the lemmatizer from spacy:
```angular2html
!python -m spacy download de_core_news_md
```

## Execution
Executing the system is easy simply call:

```angular2html
python main.py
```

In the [main.py](main.py), you can set the VECTORIZATIONTYPE flag to different types of 
vectorizers(stylometry, bag of words, tf-idf) and depending on the set value, you will receive then a different
feature matrix and will produce then results described in the report.

Changing the value FIXED_NUMBER_COMMENTS, changes how many authors with this number FIXED_NUMBER_COMMENTS 
of comments we use for further processes.

There are also additional flags, but these are most likely if you want to save the preprocessed dataframe locally, 
so the execution time reduces.

## Code Structure

