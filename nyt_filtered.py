import pandas as pd

# Chemin vers les données filtrées
filtered_nytimes_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_nytimes.csv'
filtered_nytimes_1980_2020_path = '/home/henrijn/Documents/SHS/Iran/data/filtered_nytimes_1980_2020.csv'

# Charger les données filtrées
nytimes = pd.read_csv(filtered_nytimes_path)

# Convertir la 11ème colonne en format datetime
nytimes['date'] = pd.to_datetime(nytimes.iloc[:, 10], errors='coerce')

# Filtrer les lignes pour ne garder que celles des années 1980 à 2020
filtered_nytimes_1980_2020 = nytimes[
    (nytimes['date'].dt.year >= 1980) &
    (nytimes['date'].dt.year <= 2020)
]

# Sauvegarder les données filtrées dans un nouveau fichier CSV
filtered_nytimes_1980_2020.to_csv(filtered_nytimes_1980_2020_path, index=False)

# Afficher les premières lignes des données filtrées
print(filtered_nytimes_1980_2020.head())

# Afficher le nombre d'entrées restantes après le filtrage
print(f"Number of entries: {filtered_nytimes_1980_2020.shape[0]}")

# Afficher le nombre d'entrées par année
yearly_counts = filtered_nytimes_1980_2020['date'].dt.year.value_counts().sort_index()
print("\nEntries per year:")
print(yearly_counts)

