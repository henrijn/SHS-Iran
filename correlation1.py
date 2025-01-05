import pandas as pd
import matplotlib.pyplot as plt

# Load data
hr_path = '/home/henrijn/Documents/SHS/Iran/data/human-rights-index-vdem.csv'
nation_path = '/home/henrijn/Documents/SHS/Iran/data/nation_scores_2023.csv'

hr_data = pd.read_csv(hr_path)
nation_data = pd.read_csv(nation_path)

# Filter 2023 data
hr_2023 = hr_data[hr_data.iloc[:, 2] == 2023]

# Find common countries
common_countries = set(hr_2023.iloc[:, 0]) & set(nation_data.iloc[:, 0])

# Filter both datasets for common countries
hr_filtered = hr_2023[hr_2023.iloc[:, 0].isin(common_countries)]
nation_filtered = nation_data[nation_data.iloc[:, 0].isin(common_countries)]
# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(hr_filtered.iloc[:, 3], nation_filtered.iloc[:, 1])

# Add labels for all points
for i, country in enumerate(hr_filtered.iloc[:, 0]):
    plt.annotate(country,
                (hr_filtered.iloc[i, 3],
                 nation_filtered[nation_filtered.iloc[:, 0] == country].iloc[0, 1]),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points')

plt.xlabel('Human Rights Index')
plt.ylabel('Nation Brand Index')
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('/home/henrijn/Documents/SHS/Iran/data/correlation1.png')
plt.close()

# Print correlation coefficient
correlation = pd.Series(hr_filtered.iloc[:, 3]).corr(pd.Series(nation_filtered.iloc[:, 1]))
print(f"Correlation coefficient: {correlation:.3f}")