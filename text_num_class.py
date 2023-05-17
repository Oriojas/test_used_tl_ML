import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

SEED = 137

df_results = pd.DataFrame(columns=["model", "accuracy_score",
                                   "precision_score", "recall_score",
                                   "f1_score"])

df_text = pd.read_pickle("data/df_text_clean.pkl")
df_num = pd.read_pickle("data/df_num_norm.pkl")
df_num = df_num.drop(columns=["bin_class"])

corpus = []
for i in range(len(df_text)):
    corpus.append(df_text["clean_text"].iloc[i])


cv = CountVectorizer(max_features=1500)
X_text = cv.fit_transform(corpus).toarray()
X_num = df_num.to_numpy()

y = df_text["label"].values

X = []
for i in range(len(X_num)):
    temp = np.concatenate((X_text[0], X_num[0]))
    X.append(temp)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=SEED)

# Regresión Logística
classifier = LogisticRegression(random_state=SEED)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[0] = ["LogisticRegression",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_num_class.csv")
print(df_results)



# KNN
classifier = KNeighborsClassifier(n_neighbors=5,
                                  metric="minkowski",
                                  p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

df_results.loc[1] = ["KNeighborsClassifier",
                     accuracy_score(y_test, y_pred),
                     precision_score(y_test, y_pred),
                     recall_score(y_test, y_pred),
                     f1_score(y_test, y_pred)]

df_results.to_csv("data/results_text_num_class.csv")
print(df_results)
