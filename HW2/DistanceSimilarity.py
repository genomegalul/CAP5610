import numpy as np

# Generating two random 3D vectors.
vec1 = np.random.rand(3)
vec2 = np.random.rand(3)

# Outputting the vectors.
print("Vector 1:", vec1)
print("Vector 2:", vec2)

# Finding cosine similarity.
dot_product = np.dot(vec1, vec2)
norm_vec1   = np.linalg.norm(vec1)
norm_vec2   = np.linalg.norm(vec2)
cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

# Finding Euclidean distance.
euclidean_distance = np.linalg.norm(vec1 - vec2)

# Outputting the similarity and distance measures.
print("\nCosine Similarity:", cosine_similarity)
print("Euclidean Distance:", euclidean_distance)