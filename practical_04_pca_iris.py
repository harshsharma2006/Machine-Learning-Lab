import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
iris = load_iris()
X = iris.data
y = iris.target
 
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
 
result = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
result["Species"] = [iris.target_names[i] for i in y]
print(result.head())
print("Original dimensions:", X.shape)
print("Reduced dimensions:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
 
for name in iris.target_names:
    mask = result["Species"] == name
    plt.scatter(result.loc[mask, "PC1"], result.loc[mask, "PC2"], label=name)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Iris Dataset after PCA")
plt.legend()
plt.show()
