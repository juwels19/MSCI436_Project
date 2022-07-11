import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache
def load_data():
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
    all_data = pd.concat([netflix_data, disney_data, hulu_data, amazon_data], ignore_index=True)
    #all_data = pd.concat([netflix_data, disney_data], ignore_index=True)

    # Randomize the data between each streaming platform and drop NA values
    #subset_data = all_data[['type', 'title', 'rating', 'description', 'platform']].sample(frac=1).reset_index(drop=True).dropna()
    #subset_data = disney_data[['type', 'title', 'rating', 'description', 'platform']].dropna()
    subset_data = all_data[['type', 'title', 'rating', 'description', 'platform']].sample(frac=1).reset_index(drop=True).dropna()

    return subset_data

@st.cache
def load_tfidf_matrix():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'])
    return tfidf_matrix


@st.cache
def load_cosine_similarity(tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, data, cosine_sim, indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return data[['title','description']].iloc[movie_indices]

st.title("Streaming Platform Recommendation System")

user_input = st.text_input("Movie or TV Show")

data = load_data()

tfidf_matrix = load_tfidf_matrix()

cosine_sim = load_cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

#print (get_recommendations('Grown Ups'))
results = get_recommendations(user_input, data, cosine_sim, indices)

st.table(results)

## Currently works, but you have to input a valid movie or tv show name (case sensitive)
## for results to show in a basic looking table