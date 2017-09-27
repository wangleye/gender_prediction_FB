import pymysql
import pandas as pd

conn = pymysql.connect(host= "localhost",
                  user="root",
                  passwd="123456",
                  db="all0504")

def find_users():
	query_statement = "SELECT iduser, gender, musicstr, moviestr, bookstr, televisionstr, gamestr FROM user where gender != 0 and content>0"
	users_gender = pd.read_sql_query(query_statement, conn)

	print(users_gender.sample(5))
	users_gender.to_pickle('./data/users_with_gender_interests.pkl')


def load_users():
	return pd.read_pickle('./data/users_with_gender_interests.pkl')

if __name__ == "__main__":
	users = load_users()
	print(users.describe())