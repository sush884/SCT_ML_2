# Mall Customer Segmentation using K-means Clustering

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 2: Load the Dataset
data = pd.read_csv("Mall_Customers.csv")

# Step 3: Explore Dataset
print("First 5 rows:\n", data.head())
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Step 4: Select Features (Annual Income and Spending Score)
X = data.iloc[:, [3, 4]].values

# Step 5: Elbow Method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 6: Apply K-means with optimal clusters (assume 5 from elbow method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Step 7: Visualize Clusters
plt.figure(figsize=(8,6))
plt.scatter(X[data['Cluster'] == 0, 0], X[data['Cluster'] == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[data['Cluster'] == 1, 0], X[data['Cluster'] == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[data['Cluster'] == 2, 0], X[data['Cluster'] == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[data['Cluster'] == 3, 0], X[data['Cluster'] == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[data['Cluster'] == 4, 0], X[data['Cluster'] == 4, 1], s=100, c='magenta', label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=300, c='yellow', marker='*', label='Centroids')

plt.title("Customer Segmentation using K-means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Step 8: Analyze Clusters
cluster_summary = data.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
print("\nCluster-wise Summary:\n")
print(cluster_summary)
