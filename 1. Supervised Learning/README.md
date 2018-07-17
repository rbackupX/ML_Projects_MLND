# Project to learn Supervised learning

```FIND DONORS FOR CHARITYML WITH KAGGLE```


# Import libraries necessary for this project

# Load the customers dataset


# Transforming Skewed Continuous Features
```
# Log-transform the skewed features
skewed = ['SKEWED COLUMNS',.....]
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)
```


### Normalizing Numerical Features¶

```
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['NUMERICAL COLUMNS',.....]
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))
```

### Implementation: Data Preprocessing¶
#### Perform One hot encoding on non-numeric column

```
Use pandas.get_dummies() to perform one-hot encoding on the 'features_raw' data.
Convert the target label 'income_raw' to numerical entries.
Set records with "<=50K" to 0 and records with ">50K" to 1.
```


### Shuffle and Split Data¶

```
# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 10)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
```

## Evaluating Model Performance¶
