# -*- coding: utf-8 -*-
"""Example of usage of the SilentSpeechClassifier package."""

from gridSearchParameters import *
from expVariants import *
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


# example of a classification variant. This set contains 4 prompts
# and assigns them into 2 categories. THe implementation encourages
# experimantation and divisions can be absolutely arbitral.
mode_example = {
        'gnaw' : 'category1',
        'knew' : 'category1',
        'pat' : 'category2',
        'pot' : 'category2'}
# list of experimental wariants from expVariants.
mode_list = (mode1(), mode2(), mode3(), mode4(), mode5(), mode6())

# load parameters for grid search from gridSearchParameters.
parameters_list = (para_svc(), para_knn(), para_lda())

# Full list of subjects from the study:
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
    subject.prepare_data(mode=mode_list[0], scale_data=True)
    X, Y = subject.find_best_features(feature_limit=30)

    for idx, cl in enumerate(Classifier.registry):
        cl.grid_search_sklearn(X, Y, parameters_list[idx])

    for cl in Classifier.registry:
        score = cl.classify(X, Y)

    # del subject