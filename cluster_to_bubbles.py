import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# Load data
nyt_articles_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_articles_by_cluster_themes_with_positivity_score.csv.csv'
nytimes = pd.read_csv(nyt_articles_path)

def extract_values(keywords_str):
    try:
        # Convert string to list of dicts
        keywords_list = ast.literal_eval(keywords_str)
        # Extract only 'value' fields
        values = [item['value'] for item in keywords_list]
        return values
    except:
        return []

# Apply transformation to extract all 'value' fields into a list
all_values = []
nytimes.iloc[:, 9].apply(lambda x: all_values.extend(extract_values(x)))

# Count occurrences of each 'value' field
value_counts = Counter(all_values)

# Rank fields by frequency
ranked_values = value_counts.most_common()

# Save top 100 ranked fields to CSV
ranked_values_df = pd.DataFrame(ranked_values[:100], columns=['Field', 'Frequency'])
ranked_values_df.to_csv('/home/henrijn/Documents/SHS/Iran/data/top100_ranked_fields.csv', index=False)

# Get top 100 words and their frequencies
top_words = [word for word, _ in ranked_values[:100]]
frequencies = [count for _, count in ranked_values[:100]]


# Define words to filter out
filtered_words = {'Iran',
                  'United States International Relations',
                  'United States Politics and Government',
                  'Politics and Government',
                  'United States',
                  'Presidential Election of 2020',
                  'International Relations'
                  }

# Filter top words and frequencies
filtered_indices = [i for i, word in enumerate(top_words) if word not in filtered_words]
top_words = [top_words[i] for i in filtered_indices]
frequencies = [frequencies[i] for i in filtered_indices]

# Build co-occurrence matrix
cooccurrence = np.zeros((len(top_words), len(top_words)))
word_to_index = {word: i for i, word in enumerate(top_words)}

# Build co-occurrence matrix
for _, row in nytimes.iterrows():
    row_words = extract_values(row.iloc[9])
    row_words = [w for w in row_words if w in word_to_index]
    for i, word1 in enumerate(row_words):
        for word2 in row_words[i+1:]:
            idx1, idx2 = word_to_index[word1], word_to_index[word2]
            cooccurrence[idx1, idx2] += 1
            cooccurrence[idx2, idx1] += 1

# Convert to distances (similarity to distance)
distances = 1 / (1 + cooccurrence)
np.fill_diagonal(distances, 0)

# Use MDS to get 2D coordinates
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(distances)

# Get sentiment scores and create colormap
sentiment_scores = nytimes.iloc[:, 11]
colors = [(1-score, score, 0) for score in sentiment_scores]

# Extract page numbers and dates
pages = nytimes.iloc[:, 7]
dates = nytimes.iloc[:, 3]

# Calculate weights using formula
weights = np.exp(dates - 2010) / np.log(pages)

# Create bubble chart
plt.figure(figsize=(20, 20))
sizes = weights * 100  # Scale weights for visualization

# Plot bubbles with colors
scatter = plt.scatter(coords[:, 0], coords[:, 1],
                     s=sizes,
                     alpha=0.6,
                     c=colors)

# Add labels
for i, word in enumerate(top_words):
    plt.annotate(word, (coords[i, 0], coords[i, 1]),
                fontsize=8, ha='center', va='center')

plt.axis('equal')
plt.savefig('/home/henrijn/Documents/SHS/Iran/data/word_bubbles.png', dpi=300, bbox_inches='tight')
plt.close()
