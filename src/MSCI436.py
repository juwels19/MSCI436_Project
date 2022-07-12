import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache
def load_data():
    # Read in the streaming platform data
    netflix_data = pd.read_csv('data/netflix_titles.csv').drop_duplicates()
    netflix_data["platform"] = "Netflix"
    amazon_data = pd.read_csv('data/amazon_prime_titles.csv').drop_duplicates()
    amazon_data["platform"] = "Amazon Prime"
    hulu_data = pd.read_csv('data/hulu_titles.csv').drop_duplicates()
    hulu_data["platform"] = "Hulu"
    disney_data = pd.read_csv('data/disney_plus_titles.csv').drop_duplicates()
    disney_data["platform"] = "Disney+"

    # Combine all of the data files since they have the same columns
    all_data = pd.concat([netflix_data, disney_data, hulu_data, amazon_data], ignore_index=True).drop_duplicates()
    #all_data = pd.concat([netflix_data, disney_data], ignore_index=True)

    # Randomize the data between each streaming platform and drop NA values
    #subset_data = all_data[['type', 'title', 'rating', 'description', 'platform']].sample(frac=1).reset_index(drop=True).dropna()
    #subset_data = disney_data[['type', 'title', 'rating', 'description', 'platform']].dropna()
    subset_data = all_data[['type', 'title', 'rating', 'description', 'platform']].sample(frac=1).reset_index(drop=True).dropna()

    return subset_data

@st.cache
def load_tfidf_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'])
    return tfidf_matrix


@st.cache
def load_cosine_similarity(tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, data, cosine_sim, indices):
    cpy = data.copy()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    cpy['sim_scores'] = sim_scores
    return cpy[['title', 'type', 'description', 'platform']].iloc[movie_indices]

st.title("Streaming Platform Recommendation System")

data = load_data()

tfidf_matrix = load_tfidf_matrix(data)

cosine_sim = load_cosine_similarity(tfidf_matrix)

indices = pd.Series(data.index, index=data['title']).drop_duplicates()

user_input = st.selectbox("Pick a Movie or TV Show from Netflix, Amazon Prime, Disney+, or Hulu here:", data['title'])

results = get_recommendations(user_input, data, cosine_sim, indices)
st.header(f"Use the options below to filter your results for {user_input}")
st.subheader(f"Streaming Platform(s):")
col1, col2 = st.columns(2)
netflix = col1.checkbox("Netflix", True)
amazon = col1.checkbox("Amazon Prime", True)
disney = col2.checkbox("Disney+", True)
hulu = col2.checkbox("Hulu", True)
st.subheader(f"Recommendation Type(s):")
col1, col2 = st.columns(2)
tv_show = col1.checkbox("TV Show", True)
movie = col2.checkbox("Movie", True)

if not netflix:
    results = results[results['platform'] != "Netflix"]
if not amazon:
    results = results[results['platform'] != "Amazon Prime"]
if not disney:
    results = results[results['platform'] != "Disney+"]
if not hulu:
    results = results[results['platform'] != "Hulu"]

if not tv_show:
    results = results[results['type'] != "TV Show"]
if not movie:
    results = results[results['type'] != "Movie"]

st.write("***")

count = 0
for index, row in results.iterrows():
    st.subheader(f"Rank {count+1}: {row['title']} ({row['type']} on {row['platform']})\nDescription: {row['description']}")
    count += 1
    if count == 10:
        break


# st.table(results[1:11])
