import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Read in the streaming platform data
netflix_data = pd.read_csv('data/netflix_titles.csv')
netflix_data["platform"] = "netflix"
amazon_data = pd.read_csv('data/amazon_prime_titles.csv')
amazon_data["platform"] = "amazon"
hulu_data = pd.read_csv('data/hulu_titles.csv')
hulu_data["platform"] = "hulu"
disney_data = pd.read_csv('data/disney_plus_titles.csv')
disney_data["platform"] = "disney"

# Combine all of the data files since they have the same columns
all_data = pd.concat([netflix_data, amazon_data, hulu_data, disney_data])

# Randomize the data between each streaming platform and drop NA values
subset_data = all_data[['type', 'title', 'rating', 'description', 'platform']].sample(frac=1).reset_index(drop=True).dropna()

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(subset_data['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(subset_data.index, index=subset_data['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    print(sim_scores)
    # sim_scores = sorted(sim_scores, key=lambda x: x.iat[1], reverse=True)
    # sim_scores = sim_scores[1:11]
    # movie_indices = [i[0] for i in sim_scores]
    # return subset_data[['title','description']].iloc[movie_indices]

get_recommendations('Grown Ups')