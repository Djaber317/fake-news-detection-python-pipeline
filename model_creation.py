from pretraitement_data import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier
from imblearn.metrics import classification_report_imbalanced

preprocessed_data = pretraitement_data()

#getting input texts and lables 
texts_list, labels_list = [], []

for element in preprocessed_data:
    texts_list.append(element[0])
    labels_list.append(element[1])


text_train, text_test, label_train, label_test = train_test_split(texts_list,
                                                                  labels_list,test_size=0.3)


text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm',
                          PassiveAggressiveClassifier(loss='hinge',
                                        max_iter=300,tol=1e-4, C=1.0, random_state=0))])

_ = text_clf.fit(text_train, label_train)
predicted_svm = text_clf.predict(text_test)
print(classification_report_imbalanced(label_test, predicted_svm))

np.mean(predicted_svm == label_test)
