# Natural Language Processing SS 2022
Natural language processing



``` Download stopwords first in preprocessing```



## Possible errors:
If you receive the error "en_core_web_sm" not found or something similar, please install it via
```angular2html
python3 -m spacy download en_core_web_sm
```
## Tasks

### Preprocessing

- NLP Preprocessing: stopwords removal, lematize, ...
- data Preprocessing: bring the data in a format we want






### Vectorization of data (create word embeddings)
vectorization is the process to converting text into numerical representation, which then can be used for ML

- Types:
  - Bag Of Words
  - Word2Vec
  - Normalized TF-IDF


## Ideen

Clasification Task

Classes are Users
### Features

- average comment length
- #stopwords
- lowercase / upcase writing style
- TF-IDF
- Phrasen (sentiwordnet)
- Triggered Categories (which topics does the user write comments)
- stylometric features (see stylometry)
- sentiment detaction (positive / negative comments) \

![img.png](assets/img.png)

## TODO
1) SQL query schreiben, dass wir alle user sortiert mit allen kommentare bekommen
2) 1x mit Metadata (created_at, ..) und 1x ohne. Dann im report das vergleichen

UserID | commentare | created at
-------- |------------| --------
User1   | comment1   | yy
User1   | comment2   | 3
User2   | comment1   | yy
User2   | comment2   | 3
User3   | comment1   | yy
User3   | comment2   | 3
..   | ..         | ..



## <span style="color:red">CHANGES</span>
  - <span style="color:red">wir brauchen als target ja die User ID, weil unser NN dann eine target variable y braucht.</span>
  - <span style="color:red">das heißt wir müssen unser netz so trainieren, dass es in der matrix mehrere commentare pro user hat und lernt wie die eigenschaft der kommentare des Users sind</span>
  - <span style="color:red">deshalb hab ich die struktur der feature matrix umgeschrieben</span>


3) für jeden Kommenatar die features extra generieren (average ist halt jetzt nicht mehr,sondern nur für diesen Kommentar)
4) jeder kommentar bekommt eine spalte UserID - das ist dann die "Klasse" in der dieser Kommentar fällt
5) am ende Feature matrix
      

CommentarID | text_length | comment feature 2 | comment_feature3 | User_ID
------------|-------------|-------------------|-------------------|-----------------| 
commentar1  | 234         | yy                | 5345              | 1
commentar2  | 342         | 3                 |345                | 1
commentar3  | 433         | yy                |345                | 2
commentar4  | 232         | 3                 |345                | 2

5) ML model aufsetzen - Classification tast. Welche Klasse (User_ID) fällt dieser kommentar aufgrund der eingeschaft des kommentars

   - verschiedene models probieren