import pymysql
import pickle
import numpy as np
import pandas as pd

conn = pymysql.connect(host= "localhost",
                  user="root",
                  passwd="123456",
                  db="all0504")

interest_names = ['music', 'movie', 'book', 'television', 'game']
interest_ind = {}

def save_obj(obj, name):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_users():
	query_statement = "SELECT iduser, gender, musicstr, moviestr, bookstr, televisionstr, gamestr FROM user where gender != 0 and content>0"
	users_gender = pd.read_sql_query(query_statement, conn)

	print(users_gender.sample(5))
	users_gender.to_pickle('./data/users_with_gender_interests.pkl')

def preprocessing():
	users_df = load_users_raw()
	for interest in interest_names:
		users_df[interest+'list'] = [[]]*len(users_df)
	for index, row in users_df.iterrows():
		for interest in interest_names:
			interest_items = row[interest+'str'].split(';')
			users_df.at[index, interest+'list'] = interest_items
	print(users_df.sample(5))
	users_df.to_pickle('./data/users_with_gender_interest_array.pkl')

def load_users_raw():
	return pd.read_pickle('./data/users_with_gender_interests.pkl')

def generate_interest_indication_matrix():
	dict_interest = {}
	users_df = load_users()
	for index, row in users_df.iterrows():
		gender = row['gender']
		for interest in interest_names:
			interest_items = row[interest+'list']
			for interest_i in interest_items:
				if interest_i not in dict_interest:
					dict_interest[interest_i] = {}
					dict_interest[interest_i][1] = 0
					dict_interest[interest_i][2] = 0
				dict_interest[interest_i][gender] += 1
	save_obj(dict_interest, 'interest_gender_indication')

def generate_user_interest_feature():
	users_df = load_users()
	for interest in interest_names:
		users_df[interest+'score'] = np.nan
	for index, row in users_df.iterrows():
		for interest in interest_names:
			score = []
			weight = []
			for interest_i in row[interest+'list']:
				i_s, i_w = interest_item_to_gender(interest_i)
				if i_s is not None:
					score.append(i_s)
					weight.append(i_w)
			if len(score) > 0:
				users_df.at[index, interest+'score'] = np.average(score, weights=weight)
	for interest in interest_names:
		del users_df[interest+'str']
		del users_df[interest+'list']
	print(users_df.sample(5))
	users_df.to_pickle('./data/users_with_gender_interest_score.pkl')

def load_users():
	return pd.read_pickle('./data/users_with_gender_interest_array.pkl')  # added 'musiclist', 'movielist', etc. to store interest list in the df

def load_interest_indication_dict():
	return load_obj('interest_gender_indication')

def load_users_with_interest_scores():
	return pd.read_pickle('./data/users_with_gender_interest_score.pkl')

def interest_item_to_gender(interest_i, min_count=5): # min_count: the interest whose liked user number smaller than it will not be considered
	if (interest_i not in interest_ind) or (interest_i.strip() == ''):
		return None, None
	gender1 = interest_ind[interest_i][1]
	gender2 = interest_ind[interest_i][2]
	total_liked_user_num = gender1 + gender2
	if total_liked_user_num < min_count:
		return None, None
	else:
		return gender1*1.0/total_liked_user_num, 1.0/np.log(total_liked_user_num) # besides score, also return weight 1/log(N)


if __name__ == "__main__":
	interest_ind = load_interest_indication_dict()
	generate_user_interest_feature()
