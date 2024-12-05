import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

E_path = '../knowledge_graph_embedding_module/kg_embeddings/MetaQA/best_checkpoint/E.npy'
entity_embeddings = np.load(E_path)
entity_labels = [idx for idx in range(len(entity_embeddings))]

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(entity_embeddings)

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(entity_embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
plt.colorbar(label="Cluster Label")
plt.title("2D Visualization of Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()