import pandas as pd
import random

profiles = []

for i in range(5000):

    username = "user" + str(i)

    followers = random.randint(0, 10000)
    following = random.randint(1, 5000)
    posts = random.randint(0, 500)

    verified = random.randint(0, 1)
    bio = random.randint(0, 1)
    profile_pic = random.randint(0, 1)

    username_length = len(username)
    username_digits = sum(c.isdigit() for c in username)

    ratio = round(followers / following, 2)

    fake_score = 0

    if followers < 100:
        fake_score += 1

    if following > 1000:
        fake_score += 1

    if posts < 10:
        fake_score += 1

    if verified == 0:
        fake_score += 1

    if bio == 0:
        fake_score += 1

    if profile_pic == 0:
        fake_score += 1

    if ratio < 0.10:
        fake_score += 1

    if fake_score >= 5:
        fake = 1
    else:
        fake = 0

    profiles.append([
        username,
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

df = pd.DataFrame(
    profiles,
    columns=[
        "username",
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
    ]
)

df.to_csv("dataset/instagram_profiles.csv", index=False)

print("====================================")
print("Dataset Generated Successfully")
print("Total Profiles :", len(df))
print("====================================")

print(df.head())