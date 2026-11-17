import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
 
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = StandardScaler().fit_transform(X)
 
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)
hier = AgglomerativeClustering(n_clusters=5, linkage="ward")
h_labels = hier.fit_predict(X_scaled)
 
print("K-Means silhouette score:", round(silhouette_score(X_scaled, k_labels), 3))
print("Hierarchical silhouette score:", round(silhouette_score(X_scaled, h_labels), 3))
 
plt.figure(figsize=(7, 4))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_labels)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Customer Segments")
plt.show()
 
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(8, 4))
dendrogram(Z, truncate_mode="lastp", p=20)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()
