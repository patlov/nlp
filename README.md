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

## Runtime
If you run all classifiers you can expect to wait up to ~3min, depending on whether you already use a preprocessed dataset from
the harddrive or you need to preprocess the data, then it will take about ~5min.

Pc configuration used for these times: \
CPU: AMD 5600x \
RAM: 16GB DDR4 3200 MHZ

## Code Structure

The <em>assets</em> folder contains the images we used for the report. \
The <em>dataset</em> folder contains the million corpus dataset and the database which we got from the million corpus. 
Dataset is available under https://ofai.github.io/million-post-corpus/#data-set-description \
The <em>models</em> folder contains the used classifiers and also the deep learning network we tried, but didnt use in the later stages. \
The <em>papers</em> folder contains the papers we refer to and we read to understand what is state of the art. \
The <em>preprocess</em> folder contains two files: \
The [data_preprocessing.py](preprocess/data_preprocessing.py)
contains the logic for reducing the nr. of authors to the requested comment number e.g. 1000 comments per author. \
The [nlp_preprocessing.py](preprocess/nlp_preprocessing.py) contains all the preprocessing steps applied to the comments. \
The <em>vectorization</em> folder contains the feature extraction methods we applied (stylometry, tf-idf, bag of words) in two files: 
the [vectorization.py](vectorization/vectorization.py), and [stylometry.py](vectorization/stylometry.py).
