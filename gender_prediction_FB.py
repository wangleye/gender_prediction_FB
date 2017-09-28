import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

features = ['musicscore', 'moviescore', 'bookscore', 'televisionscore', 'gamescore']

def load_users_with_interest_scores():
	return pd.read_pickle('./data/users_with_gender_interest_score.pkl')

if __name__ == "__main__":
	users_df = load_users_with_interest_scores()
	print(users_df.sample(10))
	print("total number of users: ", len(users_df))

	y = users_df['gender']
	X = users_df[features]

	lr = LogisticRegression()
	rf = RandomForestClassifier()
	svc = SVC()

	classifier_models = [lr, rf, svc]

	for model in classifier_models:
		print(cross_val_score(model, X, y, cv=5))