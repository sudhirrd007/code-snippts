```python
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
```

# Outliers

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



# Theory

https://onedrive.live.com/redir?resid=4EC15AA462FC0FE4%212087&page=Edit&wd=target%28Naive%20Bayes.one%7C3656f66a-2e75-4aca-a452-4a8c18626633%2FIMPORTANT%7Ca86b46da-d898-4ca2-a672-8c89270bdc13%2F%29


```python

```

# Important
<hr style="border:2px solid yellow"> </hr>

* Never forgot to use log while calculation
* Never forgot Laplace smoothing unless library already is not applied it 

# Basics
<hr style="border:2px solid yellow"> </hr>

mutually exclusive : a situation when two events cannot occur at same time, ex. single dice 2 and 4 at the same time  

Here, A and B = 0 

independent events is : whereas independent events occurs when one event remains unaffected by the occurrence of the other event, ex. two dices and 2 and 4 on different dices  

Here, A and B = A * B

![image.png](attachment:14528b80-c8a4-4fcd-95f5-d10ff698d4ce.png)

![image.png](attachment:36256e6b-64e6-4560-89f0-1a33876db9c7.png)

![image.png](attachment:6372525d-9449-46d4-8dc8-20601aa19514.png)

![image.png](attachment:9957d04d-6f22-4587-b9af-8630b63faf6b.png)

![image.png](attachment:248a3318-4d0f-46ec-af43-a68d0813cc58.png)

![image.png](attachment:0df149ff-8ee1-4e85-b1d9-0b99eb37d1eb.png)

![image.png](attachment:a2bb2bf8-e3e8-47ed-8310-5215b38681b9.png)

<hr style="border:2px solid green"> </hr>

# Laplace Smoothing
<hr style="border:2px solid yellow"> </hr>

![image.png](attachment:99ff8ad5-04fb-46b2-be37-63da7b31e7e1.png)

```
>> It is used to solve problem when any new feature introduce in test data
>> Because this new feature is not in training data means probability of that would be zero and all calculation would be messed up, so to solve this problem Laplace Smoothing is used
```

## Bias and Variance Tradeoff (Underfitting and Overfitting)

![image.png](attachment:27c9134f-b4e0-4473-bf2c-649c52cff061.png)

![image.png](attachment:ab38b907-2556-494b-a4cc-e8a564ae35ff.png)

![image.png](attachment:5718f368-9a61-4d7d-9b04-5ad326c5269f.png)

```
>> It is same as KNN  
>> Low k ==> Overfitting, High k ==> Underfitting 
```

# Feature Importance, Imbalanced Data

## Imbalanced Data

* Under or Over sampling  
* Use Laplace smoothing and don't change anything as alpha would work by its own (see below)

## Other

![image.png](attachment:78bceae2-3b50-4cd6-ba33-39bd5b8a3212.png)

```
>> After descending 2nd column, words/features with high values are more important features  
>> And can achieve same for column 3 after descending it
```

![image.png](attachment:521f2c08-96a8-4bb7-b411-ceab7084a264.png)

![image.png](attachment:3ec6d2bb-2b1a-426b-901b-ba5ff82e21ae.png)

```
>> See : alpha would remain same in both +ve and –ve probability measurement  
>> But it scales minor(here –ve data with only 100 data points) from 2% to 10%  
>> And scales major(here +ve data with 900 data points) from 2% to only 3.04%  
>> When data is imbalanced than "prior" probability can messed up answer
```

# Outlier 

* In naive Bayes outlier is a word/feature which occurs very less in training data  

 

* Solution to remove or optimize the outlier: 
    * Take a threshold and remove a word/feature if it occurs less than this  
        * Ex. Take threshold = 10 and if a word occurs 5 times than discards it from training data  
    * Use Laplace smoothing

# Extra

# One
<hr style="border:2px solid yellow"> </hr>

<hr style="border:1px solid red"> </hr>

# LAST

![image-20210506091715753](Naive_bayes.assets/image-20210506091715753.png)

<img src="Naive_bayes.assets/image-20210506091826447.png" alt="image-20210506091826447" style="zoom: 33%;" />




