import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV

prediction_models = ['lr', 'rf', 'knn', 'svc']

def load_users_with_prediction():
	return pd.read_pickle('./data/users_with_gender_prediction_20000.pkl')

if __name__ == "__main__":
	users_df = load_users_with_prediction().reset_index(drop=True) # re-index all the instances
	print(users_df.sample(10))
	print("total number of users: ", len(users_df))

	lr = CalibratedClassifierCV(LogisticRegression(), cv=3, method='isotonic')
	rf = CalibratedClassifierCV(RandomForestClassifier(50), cv=3, method='isotonic')
	svc = CalibratedClassifierCV(SVC(probability=True), cv=3, method='isotonic')
	risk_models = [lr, rf, svc]
	risk_model_names = ['lr', 'rf', 'svc']

	k_fold = KFold(5, random_state=0)
	for predict_model in prediction_models:

		print("============", predict_model,"============")

		is_correct_predict = (users_df[predict_model+"_predict_y"] == users_df['gender']).astype(int)
		predict_confidence = users_df[predict_model+"_predict_y_proba"]
		predict_advantage = (users_df[predict_model+"_predict_y_proba"]+1)/(2-users_df[predict_model+"_predict_y_proba"])
		has_music = users_df['musicscore'].notnull().astype(int)
		has_movie = users_df['moviescore'].notnull().astype(int)
		has_book = users_df['bookscore'].notnull().astype(int)
		has_game = users_df['gamescore'].notnull().astype(int)
		has_tv = users_df['televisionscore'].notnull().astype(int)
		risk_X = np.transpose([predict_confidence, predict_advantage, has_music, has_movie, has_book, has_game, has_tv])
		risk_y = is_correct_predict

		predict_disclosure = np.zeros((len(users_df), len(risk_models)))
		predict_disclosure_proba = np.zeros((len(users_df), len(risk_models)))

		# cross validation for each disclosure risk model
		for train_idx, test_idx in k_fold.split(risk_X, risk_y):
			for k, model in enumerate(risk_models):
				model.fit(risk_X[train_idx],risk_y[train_idx])
				predict_disclosure[test_idx, k] = model.predict(risk_X[test_idx])
				predict_disclosure_proba[test_idx, k] = model.predict_proba(risk_X[test_idx])[:, 1]
		
		# select best disclosure model
		best_model = -1
		best_f1_score = 0
		for k, model in enumerate(risk_models):
			score = f1_score(risk_y, predict_disclosure[:, k])
			print(risk_model_names[k], "f1 score:", score)
			if score > best_f1_score:
				best_f1_score = score
				best_model = k

		users_df[predict_model+'_risk_proba'] = predict_disclosure_proba[:, best_model]
	
	users_df.to_pickle('./data/users_with_risk_proba_isotonic_20000.pkl')
	print(users_df.sample(100))
