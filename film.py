import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

movies = pd.read_csv("C:/Users/RUMEYSA/Desktop/film_bugday/archive/movies.csv")
ratings = pd.read_csv("C:/Users/RUMEYSA/Desktop/film_bugday/archive/ratings.csv")

#Removing a column 'genres' from the 'movies' dataset

movies = movies.drop('genres',axis=1)
print(movies.head())
print(movies.describe())
ratings = ratings.drop('timestamp', axis=1)
print(ratings.head())
print(ratings.describe())
ratings.rating.value_counts().sort_values().plot(kind='barh')
combined_dataset = pd.merge(movies, ratings, how='left', on='movieId')
highest_rating = combined_dataset.groupby('title')[['rating']].count().nlargest(20, 'rating')
print(highest_rating)
movies_and_users = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
print(movies_and_users)
matrix_movies_users = csr_matrix(movies_and_users.values)
print(matrix_movies_users)
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
print(model.fit(matrix_movies_users))
def recommender(movie_name, data, model ):
    #eşleşme skoru, eşleşen film adı ve eşleşen film adının movies veri setindeki indeksi. Biz burada sadece indeks (idx) değerini alırız.
    idx=process.extractOne(movie_name, movies['title'])[2]
    print('Movie Selected: ',movies['title'][idx], 'Index: ',idx)
    print('Searching.....')
    distances, indices = model.kneighbors(data[idx], n_neighbors=10)
    for i in indices:
        print(movies['title'][i])
recommender('Batman', matrix_movies_users, model)       