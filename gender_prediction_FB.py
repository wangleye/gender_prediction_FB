import pymysql
import pickle
import pandas as pd

conn = pymysql.connect(host= "localhost",
                  user="root",
                  passwd="123456",
                  db="all0504")

interest_names = ['music', 'movie', 'book', 'television', 'game']

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


def load_users():
	return pd.read_pickle('./data/users_with_gender_interest_array.pkl')  # added 'musiclist', 'movielist', etc. to store interest list in the df


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



if __name__ == "__main__":
	generate_interest_indication_matrix()