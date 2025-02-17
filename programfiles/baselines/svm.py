# This file contains the implementation of baseline SVM classifier.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

labeled_reviews = pd.read_csv('../datafiles/labeled_reviews.csv')

# n-gram SVM classifier
min_n=3
max_n=5
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        analyzer='char',
        ngram_range=(min_n, max_n),
        lowercase=True,
        strip_accents='unicode'
    )),
    ('classifier', SVC(
        C=1.0,
        kernel='linear',
        gamma='auto',
        degree=3,
        probability=True
    ))
])

X = labeled_reviews['clean_content'].to_list() 
y = labeled_reviews['Privacy-related'].to_list()
test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=test_size, random_state=random_state
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
report = classification_report(y_test, y_pred)
print(report) # Report results.