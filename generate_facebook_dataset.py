import pandas as pd
import random

rows = []

for i in range(5000):

    followers = random.randint(0, 50000)
    following = random.randint(1, 50000)
    posts = random.randint(0, 5000)

    verified = random.randint(0, 1)
    bio = random.randint(0, 1)
    profile_pic = random.randint(0, 1)

    username_length = random.randint(5, 20)
    username_digits = random.randint(0, 5)

    ratio = round(followers / following, 2)

    fake = 1 if (
        followers < 100
        and following > 1000
        and posts < 10
    ) else 0

    rows.append([
        followers,
        following,
        posts,
        verified,
        bio,
        profile_pic,
        username_length,
        username_digits,
        ratio,
        fake
    ])

df = pd.DataFrame(rows, columns=[
    "followers",
    "following",
    "posts",
    "verified",
    "bio",
    "profile_pic",
    "username_length",
    "username_digits",
    "ratio",
    "fake"
])

df.to_csv(
    "dataset/facebook_profiles.csv",
    index=False
)

print("Facebook Dataset Created Successfully!")