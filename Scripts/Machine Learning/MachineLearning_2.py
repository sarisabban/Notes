import numpy as np
import pandas as pd
from sklearn import svm, utils, preprocessing, model_selection

# Import dataset
df = pd.read_csv('data.csv')

# Balance labels by randomly removing n number of examples of l label
n = 500
l = 'Alive'
idx = df.Labels[df.Labels == 'Alive'].index.tolist()
drop = np.random.choice(idx, n, replace=False)
df = df.drop(drop)

# Extract features and labels
X = df.iloc[:, 2:].to_numpy()
Y = df.iloc[:, 1].to_numpy()

# One-Hot encode Y
#Y = np.reshape(Y, (Y.shape[0], 1))
#Y = preprocessing.OneHotEncoder().fit(Y).transform(Y).toarray()

# Encode Y
Y = preprocessing.LabelEncoder().fit(Y).transform(Y)

# Shuffle Dataset
X, Y = utils.shuffle(X, Y)

# Split to Train/Test sets
X_train, X_tests, Y_train, Y_tests = model_selection.train_test_split(X, Y)

# Standerdise X
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)
X_tests = preprocessing.StandardScaler().fit(X_tests).transform(X_tests)

#--------------------------------------------------------------------------

# Machine Learning Models
ML = svm.SVC(C=1, gamma=1, random_state=0)

# Fit
Train = ML.fit(X_train , Y_train)
print(ML.score(X_train , Y_train))
print(ML.score(X_tests , Y_tests))

# Cross Validation
scores = model_selection.cross_val_score(ML, X, Y, cv=10)
print(round(np.mean(scores), 3))
