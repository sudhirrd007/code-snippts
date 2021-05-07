# Outliers


```python
""" DataFrame divided into 2 parts : inliners and outliners """
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20)
df_ = lof.fit_predict(df)
df_ = pd.Series(df_, index=df.index)
inliers = df_[df_ == 1]
oulliers = df_[df_ == -1]
```

