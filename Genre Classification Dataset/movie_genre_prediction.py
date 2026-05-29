import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv(
    "train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["ID","TITLE","GENRE","DESCRIPTION"]
)

X = train["DESCRIPTION"]
y = train["GENRE"]

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train
model = LinearSVC()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))