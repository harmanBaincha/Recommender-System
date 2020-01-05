# Content Based RS

# importing libraries
import numpy as np
import pandas as pd
import math 


#importing data

db_movies = pd.read_csv('movies.csv')
db_links = pd.read_csv('links.csv')
db_ratings = pd.read_csv('ratings.csv')
db_tags = pd.read_csv('tags.csv')

db_movies.head()
db_ratings.head()
db_tags.head()

# Computing termfrequency (tf) and document frequency (df)
r=db_tags.groupby(['movieId','tag'], as_index=False, sort=False).count()
tf = db_tags.groupby(['movieId','tag'], as_index=False, sort=False)\
         .count()\
         .rename(columns = {'userId':'tag_count_tf'})[['movieId','tag','tag_count_tf']]
tag_distinct = db_tags[['tag','movieId']].drop_duplicates()

df = tag_distinct.groupby(['tag'], as_index=False, sort=False)\
                 .count()\
                 .rename(columns = {'movieId':'tag_count_df'})[['tag','tag_count_df']]
                 
# Computing inverse document frequency (idf)

idf = np.log10(len(np.unique(db_tags['movieId'])))
df['idf'] = idf-np.log10(df['tag_count_df'])  

tf = pd.merge(tf, df, on='tag', how='left', sort=False)
tf['tf-idf'] = tf['tag_count_tf']*tf['idf']         
tf[['movieId','tag','tf-idf']].head()  



# computation of vector length    
vect_length = tf.loc[:,('movieId','tf-idf')]
vect_length['tf-idf-sq'] = vect_length['tf-idf']**2
vect_length = vect_length.groupby(['movieId'], as_index=False, sort=False)\
                   .sum()\
                   .rename(columns = {'tf-idf-sq':'tf-idf-sq-total'})[['movieId','tf-idf-sq-total']]
vect_length['vect_length'] = np.sqrt(vect_length[['tf-idf-sq-total']].sum(axis=1))

tf = pd.merge(tf, vect_length, on='movieId', how='left', sort=False)
tf['tag_vec'] = tf['tf-idf']/tf['vect_length']

tf[['movieId','tag','tf-idf','vect_length','tag_vec']].head()

# user profile
ratings_filter = db_ratings[db_ratings['rating']>=3]
user_distinct = np.unique(db_ratings['userId'])
user_tag_pref = pd.DataFrame()
i = 1

userId = 89
user_index = user_distinct.tolist().index(userId)
for user in user_distinct[user_index:user_index+1]:
    
    if i%30==0:
        print ("user: ", i , "out of: ", len(user_distinct))
            
    user_data= ratings_filter[ratings_filter['userId']==user]
    user_data = pd.merge(tf,user_data, on = 'movieId', how = 'inner', sort = False)
    user_data_itr = user_data.groupby(['tag'], as_index = False, sort = False)\
                             .sum()\
                             .rename(columns = {'tag_vec': 'tag_pref'})[['tag','tag_pref']]
    user_data_itr['user']=user
    user_tag_pref = user_tag_pref.append(user_data_itr, ignore_index=True)
    i=i+1

user_distinct = np.unique(ratings_filter['userId'])
tag_merge_all = pd.DataFrame()
i = 1
userId = 89

user_index = user_distinct.tolist().index(userId)

for user in user_distinct[user_index:user_index+1]:

    user_tag_pref_all = user_tag_pref[user_tag_pref['user']==user]
    movie_distinct = np.unique(tf['movieId'])
    j = 1
    
    for movie in movie_distinct:
        
        if j%1000==0:
            print ("movie: ", j, "out of: ", len(movie_distinct), "with user: ", i, "out of: ", len(user_distinct))
        
        tf_movie = tf[tf['movieId']==movie]
        tag_merge = pd.merge(tf_movie, user_tag_pref_all, on = 'tag', how = 'left', sort = False)
        tag_merge['tag_pref'] = tag_merge['tag_pref'].fillna(0)
        tag_merge['tag_value'] = tag_merge['tag_vec']*tag_merge['tag_pref']
        
        tag_vec_val = np.sqrt(np.sum(np.square(tag_merge['tag_vec']), axis=0))
        tag_pref_val = np.sqrt(np.sum(np.square(user_tag_pref_all['tag_pref']), axis=0))
        
        tag_merge_final = tag_merge.groupby(['user','movieId'])[['tag_value']]\
                                   .sum()\
                                   .rename(columns = {'tag_value': 'rating'})\
                                   .reset_index()
        
        tag_merge_final['rating']=tag_merge_final['rating']/(tag_vec_val*tag_pref_val)
        
        tag_merge_all = tag_merge_all.append(tag_merge_final, ignore_index=True)
        j=j+1
    
    i=i+1
tag_merge_all = tag_merge_all.sort_values(['user','rating'], ascending=False)


movies_rated = db_ratings[db_ratings['userId'] == userId]['movieId']
tag_merge_all = tag_merge_all[~tag_merge_all['movieId'].isin(movies_rated)]
tag_merge_all['user'] = tag_merge_all['user'].apply(np.int64)
tag_merge_all.head(10)