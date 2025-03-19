# %%
import pandas as pd

df = pd.read_csv('../data/msk_2024_mutations_final.csv')

# keep only the columns we need [0] + [264: -3]
df = df.iloc[:, [0] + list(range(263, len(df.columns) - 2))]
df

# %%
import numpy as np

X = df.drop(columns=["Patient"]).values  # shape: (num_patients, num_features)

# Compute intersection counts: X dot X^T gives common ones.
intersection = np.dot(X, X.T)

# Compute the row sums (number of ones per patient)
row_sums = X.sum(axis=1, keepdims=True)  # shape: (num_patients, 1)

# Compute union: union[i,j] = row_sum[i] + row_sum[j] - intersection[i,j]
union = row_sums + row_sums.T - intersection

# To avoid division by zero (if a patient has no mutations), set union=1 when union==0.
union[union == 0] = 1

# Compute the Jaccard similarity matrix.
similarity_matrix = intersection / union  # shape: (num_patients, num_patients)

# save the similarity matrix
np.savetxt('../data/msk_2024_jaccard_similarity_matrix.csv', similarity_matrix, delimiter=',')


