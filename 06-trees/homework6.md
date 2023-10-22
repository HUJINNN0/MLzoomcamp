---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: bash
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Homework

> Note: sometimes your answer doesn't match one of 
> the options exactly. That's fine. 
> Select the option that's closest to your solution.


```bash
import pandas as pd
import numpy as np
import xgboost as xgb

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
```

```bash
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.tree import export_text
```

### Dataset

In this homework, we will use the California Housing Prices from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

Here's a wget-able [link](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv):

```bash
df = pd.read_csv("housing.csv")
```

The goal of this homework is to create a regression model for predicting housing prices (column `'median_house_value'`).


### Preparing the dataset 

For this homework, we only want to use a subset of data. This is the same subset we used in homework #2.

First, keep only the records where `ocean_proximity` is either `'<1H OCEAN'` or `'INLAND'`

Preparation:

* Fill missing values with zeros.
* Apply the log tranform to `median_house_value`.
* Do train/validation/test split with 60%/20%/20% distribution. 
* Use the `train_test_split` function and set the `random_state` parameter to 1.
* Use `DictVectorizer(sparse=True)` to turn the dataframe into matrices.

```bash
df = df.loc[(df.ocean_proximity == '<1H OCEAN') | (df.ocean_proximity == 'INLAND')].reset_index(drop=True)
```

```bash
# Fill missing values with 0
df.fillna(0,inplace=True)
# Apply the log transform to `median_house_value`.
df['median_house_value'] =  df['median_house_value'].apply(lambda x : np.log1p(x))
```

```bash
df.head()
```

```bash
#Do train/validation/test split with 60%/20%/20% distribution.
#Use the `train_test_split` function and set the `random_state` parameter to 1.
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.median_house_value.copy()
y_val = df_val.median_house_value.copy()
y_test = df_test.median_house_value.copy()

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']
```

```bash
len(df_train.values), len(df_val.values), len(df_test.values)
```

```bash
# Use `DictVectorizer(sparse=True)` to turn the dataframes into matrices
train_dicts = df_train.fillna(0).to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)
```

## Question 1

Let's train a decision tree regressor to predict the `median_house_value` variable. 

* Train a model with `max_depth=1`.


Which feature is used for splitting the data?

* `ocean_proximity`<--
* `total_rooms`
* `latitude`
* `population`

```bash
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)
```

```bash
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))
```

```bash
# feature use for splitting data X[5] -> population
print(dt.feature_importances_)
plot_tree(dt);
```

## Question 2

Train a random forest model with these parameters:

* `n_estimators=10`
* `random_state=1`
* `n_jobs=-1` (optional - to make training faster)


What's the RMSE of this model on validation?

* 0.045
* 0.245 <-
* 0.545
* 0.845

```bash
# Create and train RandomForestRegressor model with specific parameters
rfr = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
# Fit model on training data
rfr.fit(X_train, y_train)
```

```bash
# Predict:
y_pred = rfr.predict(X_val)
```

```bash
print('%.3f' % mean_squared_error(y_val, y_pred, squared=False))
```

## Question 3

Now let's experiment with the `n_estimators` parameter

* Try different values of this parameter from 10 to 200 with step 10.
* Set `random_state` to `1`.
* Evaluate the model on the validation dataset.


After which value of `n_estimators` does RMSE stop improving?

- 10
- 25
- 50
- 160 <-

```bash
def get_best_num_estimatots():
    scores = []
    
    for n in range(10, 201, 10):
        rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
    
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        scores.append((n, rmse))
        
    df_scores = pd.DataFrame(scores, columns=['n_estimators', 'rmse'])
    return df_scores
```

```bash
df_scores = get_best_num_estimatots()
```

```bash
plt.plot(df_scores.n_estimators, df_scores.rmse)
```

```bash
print(df_scores.round(5))
```

## Question 4

Let's select the best `max_depth`:

* Try different values of `max_depth`: `[10, 15, 20, 25]`
* For each of these values, try different values of `n_estimators` from 10 till 200 (with step 10)
* Fix the random seed: `random_state=1`


What's the best `max_depth`:

* 10 <-
* 15
* 20
* 25

```bash
def get_best_num_estimators_and_depth():
    scores = []

    for md in [10, 15, 20, 25]:
        for n in range(10, 201, 10):
            rfr = RandomForestRegressor(n_estimators=n,
                                        max_depth=md,
                                        random_state=1,
                                        n_jobs=-1)
            rfr.fit(X_train, y_train)
    
            y_pred = rfr.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
    
            scores.append((md, n, rmse))
    
    columns = ['max_depth', 'n_estimators', 'rmse']
    df_scores = pd.DataFrame(scores, columns=columns)
    return df_scores
```

```bash
df_scores2 = get_best_num_estimators_and_depth()
```

```bash
for md in [10, 15, 20, 25]:
    df_subset = df_scores2[df_scores2.max_depth == md]
    
    plt.plot(df_subset.n_estimators, df_subset.rmse,
             label='max_depth=%d' % md)

plt.legend()
```

```bash
df_scores2.groupby("max_depth")['rmse'].agg(['mean']).reset_index().round(4)#, 'std'
```

# Question 5

We can extract feature importance information from tree-based models. 

At each step of the decision tree learning algorith, it finds the best split. 
When doint it, we can calculate "gain" - the reduction in impurity before and after the split. 
This gain is quite useful in understanding what are the imporatant features 
for tree-based models.

In Scikit-Learn, tree-based models contain this information in the
[`feature_importances_`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor.feature_importances_)
field. 

For this homework question, we'll find the most important feature:

* Train the model with these parametes:
    * `n_estimators=10`,
    * `max_depth=20`,
    * `random_state=1`,
    * `n_jobs=-1` (optional)
* Get the feature importance information from this model


What's the most important feature? 

* `total_rooms`
* `median_income`<-
* `total_bedrooms`
* `longitude`

```bash
# Creating a RandomForestRegressor model with specified parameters
rfr = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)

# Training the RandomForestRegressor model with training data
rfr.fit(X_train, y_train)
```

```bash
# Get feature importances from a RandomForestRegressor model
importances = rfr.feature_importances_
# Get the feature names
features = dv.get_feature_names_out()
# Create a DataFrame to store feature importances
feature_importance = pd.DataFrame(importances, index=features)
```

```bash
feature_importance
```

```bash
# Sort the DataFrame with feature importances in descending order
sorted_feature_importance = feature_importance.sort_values(by=0, ascending=False)

# Get the most important feature (first row in the DataFrame after sorting)
most_important_feature = sorted_feature_importance.index[0]

# Get the value of the most important feature
most_important_value = sorted_feature_importance.iloc[0, 0]
# Create a horizontal bar plot to visualize feature importances
sorted_feature_importance.plot.barh()
```

## Question 6

Now let's train an XGBoost model! For this question, we'll tune the `eta` parameter:

* Install XGBoost
* Create DMatrix for train and validation
* Create a watchlist
* Train a model with these parameters for 100 rounds:

```
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
```

Now change `eta` from `0.3` to `0.1`.

Which eta leads to the best RMSE score on the validation dataset?

* 0.3 <-
* 0.1
* Both gives same


```bash
features = dv.get_feature_names_out()
# have to remove special chars from names else XGboost complains about it
features= [i.replace("=<", "_").replace("=","_") for i in features]

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
```

```bash
def parse_xgb_output(eta):
    
    features = dv.get_feature_names_out()
    # have to remove special chars from names else XGboost complains about it
    features= [i.replace("=<", "_").replace("=","_") for i in features]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
    
    
    xgb_params = {
        'eta': eta, 
        'max_depth': 6,
        'min_child_weight': 1,
    
        'objective': 'reg:squarederror',
        'nthread': 8,
    
        'seed': 1,
        'verbosity': 1,
    }
    #watchlist = [(dtrain, 'train'), (dval, 'val')]
    #%%capture output 
    model = xgb.train(xgb_params, dtrain, num_boost_round=100)
    
    y_pred = model.predict(dval)
    
    print('%.3f' % mean_squared_error(y_val, y_pred, squared=False))
    #s = output.stdout
    #return y_pred

```

```bash
parse_xgb_output(eta=0.3)
```

```bash
parse_xgb_output(eta=0.1)
```

```bash

```

## Submit the results

- Submit your results here: TBA
- If your answer doesn't match options exactly, select the closest one.
- You can submit your solution multiple times. In this case, only the last submission will be used

## Deadline

The deadline for submitting is October 23 (Monday), 23:00 CET. After that the form will be closed.
