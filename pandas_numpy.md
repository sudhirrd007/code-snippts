# Basic Operations

## Multiline Comment


```python
def MultiResults(flag=True):
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all" if flag==True else "last_expr"
MultiResults(True)
```

## Rename


```python
df.rename(columns = {'test':'TEST'}, inplace = True) 
```

## Replace


```python
df["workclass"].replace("?", np.NaN, inplace=True)
```

## is null


```python
df.isnull().sum()
```

## Combine
while combing series, first set index of all series same
## To csv


```python
df.to_csv("test.csv", index=False)
```

# Intermediate

## format of values(DataFrame)


```python
pd.options.display.float_format = '{:,.3f}'.format

def fn(x):
      return int(x) if(x%1 == 0) else "{:.2f}".format(x)
dg.style.format(fn)
```


```python

```


```python

```


```python

```

# KNN


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

### it calculate different metrics on different numbers of k values
def dist_metrics(metrics=None, ks=None):
    
    if(not metrics):
        metrics = ['canberra', 'braycurtis', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean']
    if(not ks):
        ks = [v for v in range(5,80,2)]
        
    final = {"k":ks}

    for m in tqdm(metrics):
        final[m] = []
        for k in ks:
            knn = KNeighborsClassifier(k, metric=m);
            scores = cross_val_score(knn, X, Y, cv=4)
            final[m].append(scores.mean())
    return pd.DataFrame(final)
# final = dist_metrics(X, Y)

#>>> final.mean(axis=0)


### It show line graph, comparing different metrics in the context of different score 
# acording to k values
def show_metrics(final):
    colors = np.random.rand(1, 4)

    for d in final.columns.drop("k"):
        plt.plot(np.arange(38), final[d], label=d)

    plt.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    plt.show();
# show_metrics(final)
```


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

### if calculate optimal value of k(from given list of k) for each random state(
# from given list of random state)
def random_state_wise_k(X, Y, random_state=None, ks=None, metric="canberra"):
    if(not random_state):
        random_state = [s for s in range(100)]

    final = {"random_state": random_state, "k":[], "score":[]}
    ks = [k for k in range(5,80,2)]

    for r in tqdm(random_state):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=r);
        scores = []
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric);
            knn.fit(X_train, Y_train);
            scores.append(knn.score(X_test, Y_test))
        final["k"].append(ks[scores.index(max(scores))])
        final["score"].append(max(scores))
    final = pd.DataFrame(final).sort_values(by="score", ascending=False)
    return final
# final = random_state_wise_k(X, Y)

#>>> final.k.value_counts()
```


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

### It will calculate score of every givn k values for every given random state
def k_wise_scores(X, Y, ks=None, metric="canberra"):
    if(not ks):
        ks = [k for k in range(5,80,2)]

    final = {}
    for k in ks:
        final[k] = []

    for r in tqdm(range(100)):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=r);
        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric);
            knn.fit(X_train, Y_train);
            final[k].append(knn.score(X_test, Y_test))
    final = pd.DataFrame(final)
    return final
# final = k_wise_scores(X, Y)


### it extracts best k value from each k value columns
def best_k_randomstate_match(final):
    best_match = { "score":[], "k":[], "random_state":[]}

    for i in final:
        best_match["score"].append(final[i].max())
        best_match["k"].append(i)
        best_match["random_state"].append(final[i].idxmax())

    return pd.DataFrame(final_1).sort_values(by="score", ascending=False)
# best_match = best_k_randomstate_match(final)

#>>> best_match.random_state.value_counts()
```

# user defined metrics


```python
def calculate(X, Y):
    return np.sum(np.subtract(X, Y)**6)

#>>> knn = KNeighborsClassifier(n_neighbors=13, metric=calculate)
```

# Oversampling and Undersampling


```python
from sklearn.utils import resample

def data_sampling(X_train, Y_train, sampling_type=None):
    XY_train = pd.concat([X_train, Y_train], axis=1)
    
    XY_train_0 = XY_train[XY_train["Outcome"] == 0]
    XY_train_1 = XY_train[XY_train["Outcome"] == 1]
    
    if(sampling_type == "over"):
        XY_train_1_oversampled = resample(XY_train_1, n_samples=XY_train_0.shape[0], replace=True, random_state=0)
        combined = pd.concat([XY_train_0, XY_train_1_oversampled])
    elif(sampling_type == "under"):
        XY_train_0_undersampled = resample(XY_train_0, n_samples=XY_train_1.shape[0], replace=True, random_state=0)
        combined = pd.concat([XY_train_0_undersampled, XY_train_1])
    else:
        print("Provide {sampling_type} parameter")
    
    X_train = combined.iloc[:, :-1]
    Y_train = combined.iloc[:, -1]
    return X_train, Y_train
# X_train, Y_train = data_sampling(X_train, Y_train, sampling_type="over")
```

# Matplotlib


```python
import matplotlib.pyplot as plt
plt.style.available
['Solarize_Light2',
 '_classic_test_patch',
 'bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn',
 'seaborn-bright',
 'seaborn-colorblind',
 'seaborn-dark',
 'seaborn-dark-palette',
 'seaborn-darkgrid',
 'seaborn-deep',
 'seaborn-muted',
 'seaborn-notebook',
 'seaborn-paper',
 'seaborn-pastel',
 'seaborn-poster',
 'seaborn-talk',
 'seaborn-ticks',
 'seaborn-white',
 'seaborn-whitegrid',
 'tableau-colorblind10'];
```

## Half Heatmap


```python
corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

fig = plt.gcf()
fig.set_size_inches(8, 5)
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
```

## Bar chart


```python
def bar_chart(data):
    unique = data.value_counts()

    colors = np.random.rand(len(unique), 3)
    fig = plt.gcf()
    fig.set_size_inches(3, 4)

    ax = unique.plot(kind='bar', fontsize=13, color=colors)
    plt.ylim(0, data.shape[0])
    plt.xticks(rotation=0, fontsize=15)

    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), size=15)
    plt.show()
```

# Feature Correlation


```python
def feature_correlation(corr):
    count = 1
    index = 0
    final = pd.DataFrame(columns=["column", "row", "value"])
    

    for column in corr.columns:
        for row in list(corr.columns)[count:]:
            final.loc[index] = [column, row, corr[row][column]]
            index += 1
        count += 1
    return final.sort_values(by="value", ascending=False)
```


```python

```
