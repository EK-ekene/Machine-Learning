import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


breast_cancer=load_breast_cancer()

print(breast_cancer)

breast_cancer_df=pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)

breast_cancer_df.head()
breast_cancer_df.tail()
breast_cancer_df.info()

breast_cancer_df['label']=breast_cancer_df.target
breast_cancer_df.isnull().sum()

breast_cancer_df.describe()
breast_cancer_df.shape

breast_cancer_df['label'].value_counts()

breast_cancer_df.groupby('label').mean()

X = breast_cancer_df.drop(columns='label',axis=1)
Y = breast_cancer_df['label']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


#Building the Neural Network
#importing Tensorflow and Keras

import tensorflow as tf
tf.random.set_seed(3)

from tensorflow import keras

#setting up the neural network


