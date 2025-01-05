import pandas as pd
import numpy as np
from openai import OpenAI
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import ast

# Initialize OpenAI client
client = OpenAI()

# Load data
nyt_articles_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_nytimes.csv'
nytimes = pd.read_csv(nyt_articles_path)

def extract_values(keywords_str):
    try:
        # Convert string to list of dicts
        keywords_list = ast.literal_eval(keywords_str)
        # Extract only 'value' fields
        values = [item['value'] for item in keywords_list]
        return ' '.join(values)
    except:
        return ''

# Apply transformation
nytimes['combined_text'] = nytimes.iloc[:, 9].apply(extract_values).fillna('')

# Get embeddings using OpenAI
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None

# Get embeddings for all texts with error handling
embeddings = []
for text in nytimes['combined_text'].tolist():
    embedding = get_embedding(text)
    if embedding is not None:
        embeddings.append(embedding)
embeddings = np.vstack(embeddings)

# Normalize embeddings
embeddings_normalized = normalize(embeddings)

# Calculate cosine distances (keep as 2D matrix)
distances = cosine_distances(embeddings_normalized)

# Find optimal eps using k-distance graph
def find_optimal_eps(embeddings, n_neighbors=15):
    distances = cosine_similarity(embeddings)
    distances = 1 - distances  # Convert to distances
    distances.sort()
    distances = distances[:, 1:n_neighbors+1]  # Exclude self-distance
    mean_distances = np.mean(distances, axis=1)
    mean_distances.sort()

    knee_locator = KneeLocator(
        range(len(mean_distances)),
        mean_distances,
        curve='convex',
        direction='increasing'
    )

    plt.plot(mean_distances)
    plt.axvline(x=knee_locator.knee, color='r', linestyle='--')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Mean k-nearest neighbor distance')
    plt.savefig('kdistance_plot.png')
    plt.close()

    return mean_distances[knee_locator.knee]


# Apply HDBSCAN with precomputed distances
clusterer = HDBSCAN(
    min_cluster_size=3,
    min_samples=2,
    metric='precomputed',
    cluster_selection_epsilon=0.05
)

# Fit with full distance matrix
nytimes['cluster'] = clusterer.fit_predict(distances)

# Process each cluster with GPT-4
cluster_themes = {}
for cluster in sorted(set(nytimes['cluster'])):
    if cluster == -1:
        continue

    # Get all texts from the current cluster
    cluster_texts = nytimes[nytimes['cluster'] == cluster]['combined_text'].tolist()
    combined_keywords = ' '.join(cluster_texts)

    # Ask GPT-4 about main theme
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are analyzing keywords from news articles."},
                {"role": "user", "content": f"What is the most emerging thematic from all these keywords? Keywords: {combined_keywords}"}
            ]
        )
        theme = completion.choices[0].message.content
        cluster_themes[cluster] = {
            'size': len(cluster_texts),
            'keywords': combined_keywords[:200] + '...',  # Preview of keywords
            'theme': theme
        }
    except Exception as e:
        print(f"Error processing cluster {cluster}: {e}")

# Save themes to CSV
themes_df = pd.DataFrame.from_dict(cluster_themes, orient='index')
themes_df.to_csv('/home/henrijn/Documents/SHS/Iran/data/cluster_themes.csv')
