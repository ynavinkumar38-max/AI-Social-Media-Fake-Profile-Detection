import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(dataset_path, model_path, platform):

    print(f"\n========== {platform} ==========\n")

    data = pd.read_csv(dataset_path)

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

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{platform} Accuracy : {accuracy*100:.2f}%")

    joblib.dump(model, model_path)

    print(f"{platform} Model Saved Successfully!")

train_model(
    "dataset/instagram_profiles.csv",
    "model/instagram_model.pkl",
    "Instagram"
)
train_model(
    "dataset/facebook_profiles.csv",
    "model/facebook_model.pkl",
    "Facebook"
)