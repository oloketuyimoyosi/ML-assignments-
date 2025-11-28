# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# 1. Load the data
# I am using the direct link to the csv file
url = "https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv"
dataset = pd.read_csv(url)

# 2. Select the features we want to use
# We need Annual Income and Spending Score. 
# We can also use Age.
X = dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Scale the data
# This puts all numbers on the same scale so big numbers don't dominate
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means Clustering
# We are guessing there are 5 groups
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 5. Hierarchical Clustering
model = AgglomerativeClustering(n_clusters=5)
hierarchical_labels = model.fit_predict(X_scaled)

# 6. Compare Results
# Calculate the silhouette score for both
k_score = silhouette_score(X_scaled, kmeans_labels)
h_score = silhouette_score(X_scaled, hierarchical_labels)

print("Scores:")
print("K-Means Score:", f"{k_score*100}%")
print("Hierarchical Score:", f"{h_score*100}%")

# 7. Visualization with t-SNE
# This helps us plot 3 variables on a 2D graph
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot K-Means results
plt.figure(figsize=(6, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()

# Plot Hierarchical results
plt.figure(figsize=(6, 4))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.show()