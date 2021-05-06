# Contents

* [Categorical / Numerical Columns](#Categorical-/-Numerical-Columns) <br>
* [Missing](#Missing)

# Categorical / Numerical Columns


```python
categorical = [var for var in df.columns if df[var].dtype=='O']

numerical = [var for var in df.columns if df[var].dtype!='O']
```

<hr style="border:1px solid red"> </hr>

# Missing
[Contents](#Contents)


```python
""" Missing count(Columnwise) """
def missing_count(df):
    L = len(df)
    ms = df.isnull().sum()
    ms = pd.DataFrame(ms, columns=["Missing ("+str(L)+")"])
    ms["Missing(%)"] = round(ms / L * 100, 2)
    ms["Dtypes"] = df.dtypes
    return ms.sort_values(by="Missing(%)", ascending=False)
```


```python
""" Missing Dataset(True/False) """
null_df = df[df.isnull().any(axis=1)]
```

<hr style="border:1px solid red"> </hr>

# Class Percent


```python
def class_percent(Y):
    L = len(Y)
    values = Y.value_counts()
    df = {"Value":[], "Total":[], "Count":[], "Count(perc)":[]}
    for val,count in values.items():
        df["Value"].append(val)
        df["Total"].append(L)
        df["Count"].append(count)
        df["Count(perc)"].append(round(count/L*100, 3))
    df = pd.DataFrame(df)
    return df
```

<hr style="border:1px solid red"> </hr>

# Unique Labels


```python
def unique_lebels(X, columns=None):
    if(not columns):
        columns = X.columns
    df = {"name": columns, "total":[len(X)]*X.shape[1], "unique_labels":[]}
    for i in columns:
        df["unique_labels"].append(X[i].unique().size)
    df = pd.DataFrame(df).sort_values(by="unique_labels", ascending=False)
    return df
```

<hr style="border:1px solid red"> </hr>

# Sampling

## Over Sampling


```python
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
```

## Under Sampling


```python
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority')
X_under, y_under = undersample.fit_resample(X, y)
```

<hr style="border:1px solid red"> </hr>

# Feature Scaling

## Normalization

Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1. It is also known as Min-Max scaling

Normalization is good to use when you know that the distribution of your data does not follow a Gaussian distribution. This can be useful in algorithms that do not assume any distribution of the data like K-Nearest Neighbors and Neural Networks


```python
from sklearn.preprocessing import MinMaxScaler

norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
```

## Standardization

Standardization is another scaling technique where the values are centered around the mean with a unit standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution has a unit standard deviation

Standardization, on the other hand, can be helpful in cases where the data follows a Gaussian distribution. However, this does not have to be necessarily true. Also, unlike normalization, standardization does not have a bounding range. So, even if you have outliers in your data, they will not be affected by standardization


```python
from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(X_train_stand)
X_train_scale = scale.transform(X_train)
X_test_scale = scale.transform(X_test)
```

## RobustScaler


```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

<hr style="border:1px solid red"> </hr>

# Null Accuracy (Classification)


```python
maxi = y_train.value_counts().idxmax()
null_acc = y_test.value_counts()[maxi] / y_test.shape[0] * 100
```


```python

```
