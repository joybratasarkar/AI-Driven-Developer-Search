import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Set a random seed for reproducibility
np.random.seed(42)

# Load the MovieLens 100K dataset
ratings_url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat"
movies_url = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat"
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(ratings_url, sep='::', names=column_names, engine='python')
movies = pd.read_csv(movies_url, sep='::', names=['movie_id', 'title', 'genres'], engine='python')

# Merge ratings and movies dataframes on 'movie_id'
data = pd.merge(ratings, movies, on='movie_id')

# Limit the number of users and movies to reduce matrix size
num_users = 10000  # Limit to 1000 users
num_movies = 10000  # Limit to 1000 movies

selected_users = data['user_id'].unique()[:num_users]
selected_movies = data['title'].unique()[:num_movies]

data_filtered = data[data['user_id'].isin(selected_users) & data['title'].isin(selected_movies)]

# Create a user-item interaction matrix
user_item_matrix = data_filtered.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Apply SVD to the user-item matrix
svd = TruncatedSVD(n_components=50, random_state=42)
matrix_svd = svd.fit_transform(user_item_matrix)

# Compute the cosine similarity between items (movies)
item_similarity = cosine_similarity(svd.components_.T)

# Function to get movie recommendations based on item similarity
def get_movie_recommendations(movie_title, similarity_matrix, titles, top_n=10):
    # Find the index of the movie in the titles
    movie_idx = titles.index(movie_title)
    
    # Get similarity scores for this movie with all other movies
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort by similarity score
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top N similar movies
    top_movies = [titles[i] for i, _ in similarity_scores[1:top_n+1]]  # Skip the first one (it's the movie itself)
    
    return top_movies

# Helper function to search for movie titles
def search_movie_title(search_term, titles):
    return [title for title in titles if search_term.lower() in title.lower()]

# Check available titles
titles = user_item_matrix.columns.tolist()
print("Sample movie titles from the dataset:")
for title in titles[:10]:
    print(title)

# Search for the correct title
search_results = search_movie_title("Deadpool", titles)
for title in search_results:
    print(title)

# Example usage with a correct title
if search_results:
    movie_title = search_results[0]  # Use the first result found
    recommended_movies = get_movie_recommendations(movie_title, item_similarity, titles, top_n=5)
    print(f"Movies similar to '{movie_title}':\n")
    for i, movie in enumerate(recommended_movies, 1):
        print(f"{i}. {movie}")
else:
    print("No matching movie found.")
