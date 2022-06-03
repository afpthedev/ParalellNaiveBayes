# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:58:53 2022

@author: yediz
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# Seaborn is a library for making statistical graphics in Python.
import seaborn as sns
sns.set_style('darkgrid')



# Loading Datasets

data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')
data.head(10)

# # electronic design automation: EDA 
# data['diagnosis'].hist()

# #Heatmapping  for datasets
# corr = data.iloc[:,:-1].corr(method="pearson")
# cmap = sns.diverging_palette(250,354,80,60,center='dark',as_cmap=True)
# sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)

# #Data
data = data[["mean_radius", "mean_texture", "mean_smoothness", "diagnosis"]]
# data.head(10)

# # histograms   ----- > ylabels = ' Count' xlabels = 'radius_mean,smoothness_mean,texture_mean'
# fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
# sns.histplot(data, ax=axes[0], x="mean_radius", kde=True, color='r')
# sns.histplot(data, ax=axes[1], x="mean_smoothness", kde=True, color='b')
# sns.histplot(data, ax=axes[2], x="mean_texture", kde=True)


# Calculate P(Y=y) for all possible y 

def calculate_prior(df,Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes: 
        prior.append(len(df[df[Y]==i])/len(df))
    return prior    

# Approch 1 : P(X=x|Y=y) using Gaussian dist.

def calculate_likelihood_gaussian(df,feat_name,feat_val,Y,label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean,std = df[feat_name].mean(),df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y

# Calculate P(X= x1 | Y= y )P(X= x1 | Y= y )P(X= x2 | Y= y )P(X= x3 | Y= y )P(X= xn | Y= y )  * P(Y|y)
def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 
    
# Test Gaussian model

# from sklearn.model_selection import train_test_split
# train, test = train_test_split(data, test_size=.2, random_state=41)

# X_test = test.iloc[:,:-1].values
# Y_test = test.iloc[:,-1].values
# Y_pred = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")

# from sklearn.metrics import confusion_matrix, f1_score
# print(confusion_matrix(Y_test, Y_pred))
# print(f1_score(Y_test, Y_pred))

# data["cat_mean_radius"] = pd.cut(data["mean_radius"].values, bins = 3, labels = [0,1,2])
# data["cat_mean_texture"] = pd.cut(data["mean_texture"].values, bins = 3, labels = [0,1,2])
# data["cat_mean_smoothness"] = pd.cut(data["mean_smoothness"].values, bins = 3, labels = [0,1,2])

# data = data.drop(columns=["mean_radius", "mean_texture", "mean_smoothness"])
# data = data[["cat_mean_radius",	"cat_mean_texture",	"cat_mean_smoothness", "diagnosis"]]
# data.head(10)

# Approach 2: Calculate P(X=x|Y=y) categorically
 
def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y


# Calculate P(X=x1|Y=y)P(X=x2|Y=y)...P(X=xn|Y=y) * P(Y=y) for all y and find the maximum
def naive_bayes_categorical(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j] # üst kısım kodlanıyor 

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 

import time 

# Test Categorical model 
starttime = time.time()
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=.2, random_state=41)
    
X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_categorical(train, X=X_test, Y="diagnosis")
    
from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(Y_test, Y_pred))
print('Confusion matris Result')
print(1- f1_score(Y_test, Y_pred))
end_time  = time.time() - starttime
print(f"Processing {len(data)} numbers took {end_time} time using serial processing.")
print()
    
