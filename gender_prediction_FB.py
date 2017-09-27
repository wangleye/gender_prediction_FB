import pymysql

conn = pymysql.connect(host= "localhost",
                  user="root",
                  passwd="123456",
                  db="all0504")

workEdu = {}
work_cc = {}
def find_users(query_types):
	query_str = ', '.join(query_types)
	query_statement = "SELECT iduser, gender, musicstr, moviestr, bookstr, televisionstr, gamestr FROM user where gender != 0 and content>0".format(query_str)
	x = conn.cursor()
	x.execute(query_statement)
	results = x.fetchall()
	print 'start count results...'
	for result in results:
		user_id = result[0]
		if user_id not in u_cc_map:
			continue
		
		### store all the we
		result_str = ''
		for idx in range(1, len(query_types)):
			if result[idx].strip() == '' or result[idx].strip() == '""':
				continue
			result_str += result[idx] + ','
		if result_str == '':
			continue
		recent_we = result_str.split(',')[0]
		workEdu[user_id] = recent_we
		if recent_we not in work_cc:
			work_cc[recent_we] = set()
		work_cc[recent_we].add(u_cc_map[user_id])

		#### only store recent we
		# if entities[0] not in workEdu: # only keep the recent work/education
		# 	workEdu[entities[0]] = 1
		# else:
		# 	workEdu[entities[0]] += 1