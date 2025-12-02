import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column (species)
df['target'] = iris.target

# Optional: map target numbers to species names
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

print(df.head())