import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

SEED = 137

df_results = pd.DataFrame(columns=["model", "accuracy_score",
                                   "precision_score", "recall_score",
                                   "f1_score"])

df = pd.read_pickle("data/df_text_clean.pkl")

corpus = []
for i in range(len(df)):
    corpus.append(df["clean_text"].iloc[i])


cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=SEED)

# clasificador Gausiano
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[0] = ["GaussianNB",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_class.csv")
print(df_results)


# Regresión Logística
classifier = LogisticRegression(random_state=SEED)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[1] = ["LogisticRegression",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_class.csv")
print(df_results)


# KNN
classifier = KNeighborsClassifier(n_neighbors=5,
                                  metric="minkowski",
                                  p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[2] = ["KNeighborsClassifier",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_class.csv")
print(df_results)


# SVM k_lineal
classifier = SVC(kernel="linear",
                 random_state=SEED)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[3] = ["SVM_k_lineal",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_class.csv")
print(df_results)

# SVM k_sigmoid
classifier = SVC(kernel="sigmoid",
                 random_state=SEED)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[4] = ["SVM_sigmoid",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]
df_results.to_csv("data/results_text_class.csv")
print(df_results)
