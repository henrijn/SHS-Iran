import pandas as pd
import matplotlib.pyplot as plt

# Load data
nation_path = '/home/henrijn/Documents/SHS/Iran/data/nation_scores_2023.csv'
values_path = '/home/henrijn/Documents/SHS/Iran/data/country_values.csv'

nation_data = pd.read_csv(nation_path)
values_data = pd.read_csv(values_path)

# Find common countries
common_countries = set(nation_data.iloc[:, 0]) & set(values_data.iloc[:, 0])

# Filter both datasets for common countries and sort
nation_filtered = nation_data[nation_data.iloc[:, 0].isin(common_countries)].sort_values(by=nation_data.columns[0])
values_filtered = values_data[values_data.iloc[:, 0].isin(common_countries)].sort_values(by=values_data.columns[0])

# Create scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(values_filtered.iloc[:, 1], nation_filtered.iloc[:, 1])

# Add labels for all points
for i in range(len(nation_filtered)):
    country = nation_filtered.iloc[i, 0]
    x = values_filtered.iloc[i, 1]
    y = nation_filtered.iloc[i, 1]
    plt.annotate(country,
                (x, y),
                fontsize=8,
                xytext=(5, 5),
                textcoords='offset points')

plt.xlabel('Global Peace Index')
plt.ylabel('Nation Brand Index')
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('/home/henrijn/Documents/SHS/Iran/data/correlation2.png')
plt.close()

# Print correlation coefficient and data
correlation = pd.Series(values_filtered.iloc[:, 1]).corr(pd.Series(nation_filtered.iloc[:, 1]))
print(f"Correlation coefficient: {correlation:.3f}")

# Print detailed country information
print("\nCountry Details:")
print(f"{'Country':<30} {'Values':>10} {'Nation Brand':>12}")
print("-" * 55)
for i in range(len(nation_filtered)):
    country = nation_filtered.iloc[i, 0]
    values_score = values_filtered.iloc[i, 1]
    brand_score = nation_filtered.iloc[i, 1]
    print(f"{country:<30} {values_score:>10.3f} {brand_score:>12.3f}")