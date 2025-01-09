import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, RobustScaler 
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.patches as mpatches
import time
import os


# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('creditcard.csv')  

print("There are", df["Class"].value_counts()[0]/len(df)*100,"% non-fradulent transactions")
print("There are",df["Class"].value_counts()[1]/len(df)*100, "% fradulent transaction")

colors = ["#0101DF", "#DF0101"]

sns.countplot(x='Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()

standard_scaler= StandardScaler()
robust_scaler= RobustScaler()


df['scaled_amount'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

print("Scaling data... \n", df.head(10))

features = df.drop('Class', axis=1)
target = df['Class']

stratified = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in stratified.split(features, target):
    print("Train:", train_index, "Test:", test_index)
    original_featuretrain, original_featuretest = features.iloc[train_index], features.iloc[test_index]
    original_targettrain, original_targettest = target.iloc[train_index], target.iloc[test_index]




# Turn into an array
original_featuretrain = original_featuretrain.values
original_featuretest = original_featuretest.values
original_targettrain = original_targettrain.values
original_targettest = original_targettest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_targettrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_targettest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_targettrain))
print(test_counts_label/ len(original_targettest))


df = df.sample(frac=1)


fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=7)

print(new_df.head())


print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))



sns.countplot(x='Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()