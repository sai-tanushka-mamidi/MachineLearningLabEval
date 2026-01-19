#Q6
import numpy as np
import pandas as pd
df_encoded = pd.get_dummies(df)
vec1 = df_encoded.iloc[0].values
vec2 = df_encoded.iloc[1].values
cosine_similarity = np.dot(vec1, vec2) / 
    (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print("Cosine Similarity:", cosine_similarity)
