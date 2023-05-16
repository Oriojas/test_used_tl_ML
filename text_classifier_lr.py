import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("data/df_text_train.csv")
# Load the dataset
texts = list(df["clean_text"])  # list of text samples
labels = list(df["label"])  # list of corresponding labels (0 or 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=137)

# Convert the text to a bag-of-words representation
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
