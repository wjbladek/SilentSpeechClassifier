# -*- coding: utf-8 -*-
"""Example of usage of the SilentSpeechClassifier module."""

from gridSearchParameters import *
from SSC import Dataset, Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

PATH_TO_DATA = "YOUR_PATH_TO_KARA ONE/p/"

# Build instances of classifiers.
Classifier('Support Vector Classifier', SVC())
Classifier("k-nearest neighbors", KNeighborsClassifier())
Classifier("Linear Discriminant Analysis", LDA())

# load parameters for grid search
parameters_list = (para_svc(), para_knn(), para_lda())

# SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')
SUBJECTS = ('MM05',)

# initilise subjects's instances.
for subject in SUBJECTS:
    Dataset(subject)

# Iterate over subjects, preprocess the data and get scores.
for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=True)
    subject.select_channels(channels=5)
    subject.filter_data(lp_freq=None, hp_freq=1, save_filtered_data=False, plot=False)
    subject.prepare_data(mode=2, scale_data=True)
    X, Y = subject.find_best_features(feature_limit=30, statistic='Anova')

    for idx, cl in enumerate(Classifier.registry):
        cl.grid_search_sklearn(X, Y, parameters_list[idx])

    for cl in Classifier.registry:
        score = cl.classify(X, Y)

    # del subject