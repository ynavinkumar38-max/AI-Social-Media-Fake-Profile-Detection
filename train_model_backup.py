import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data = pd.read_csv("dataset/instagram_profiles.csv")

print("Dataset Loaded Successfully")
print(data.head())

X = data[
    [
        "followers",
        "following",
        "posts",
        "verified",
        "bio",
        "profile_pic",
        "username_length",
        "username_digits",
        "ratio"
    ]
]

y = data["fake"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Accuracy : {accuracy*100:.2f}%")
print("==============================")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance\n")
print(
    importance.sort_values(
        by="Importance",
        ascending=False
    )
)

joblib.dump(model, "model/model.pkl")

print("\nModel Saved Successfully!")