import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

features = ['musicscore', 'moviescore', 'bookscore', 'televisionscore', 'gamescore']

def load_users_with_interest_scores():
	return pd.read_pickle('./data/users_with_gender_interest_score_10000.pkl')

if __name__ == "__main__":
	users_df = load_users_with_interest_scores()
	print(users_df.sample(10))
	print("total number of users: ", len(users_df))

	y = np.array(users_df['gender'])
	X = np.array(users_df[features])

	lr = LogisticRegression()
	rf = RandomForestClassifier(100)
	svc = SVC(probability=True)
	knn = KNeighborsClassifier(50)

	# classifier_models = [lr, rf, knn, svc]
	classifier_models = [lr, rf, knn, svc]
	classifier_names = ['lr', 'rf', 'knn', 'svc']
	
	predict_y = np.zeros((len(users_df), len(classifier_models)))
	predict_y_proba = np.zeros((len(users_df), len(classifier_models)))

	k_fold = KFold(5)
	for train_idx, test_idx in k_fold.split(X, y):
		for k, model in enumerate(classifier_models):
			model.fit(X[train_idx],y[train_idx])
			predict_y[test_idx, k] = model.predict(X[test_idx])
			predict_y_proba[test_idx, k] = np.max(model.predict_proba(X[test_idx]), axis=1)

	for k, model in enumerate(classifier_models):
		users_df[classifier_names[k]+'_predict_y'] = predict_y[:, k]
		users_df[classifier_names[k]+'_predict_y_proba'] = predict_y_proba[:, k]

	print(users_df.sample(10))
	users_df.to_pickle('./data/users_with_gender_prediction.pkl')

	

