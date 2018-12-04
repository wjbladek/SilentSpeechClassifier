# -*- coding: utf-8 -*-
# version 1.1
"""Preproccessing and machine learning tools for the Kara One database
(openly available EEG data from an imagined speech research). 
"""

import glob, os
import mne
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import signal as sig
from time import time

import aseegg as aseegg
from features import fast_feat_array


class Dataset:
    """Loading and preprocessing of the KARA ONE database.

    Notes
    -----
    The class provides means for easily automated preproccessing 
    and preparation of the KARA ONE dataset. Most of the data
    is kept within the class, only two methods
    (prepare_data() and find_best_features()) are not void.

    Attributes
    ---------
    registry : list
        list of class instances.
    subject : string
        subject name, in this case 'MM05', 'MM21' etc.
        Used for file navigation.
    figures_path : str
        By default it is /YOUR_SCRIPT/figures,
        folder is created if there is none.

    Methods
    -------
    load_data(path_to_data, raw=True)
        Load subject data from the Kara One dataset.
    select_channels(channels=5)
        Choose how many or which channels to use.
    filter_data(lp_freq=49, hpfreq=1, save_filtered_data=False, plot=False)
        Filter subject's EEG data.
    ica(ica='fast')
        Exclude components using Independent Component Analysis.
    prepare_data(mode=2)
        Organise subject's EEG data for machine learning.
    find_best_features(feature_limit=30, scale_data=True, statistic='Anova')
        Select n best features.

    Examples
    --------

    import IspeechInterface
    SUBJECTS = ('MM05',)
    for subject in SUBJECTS:
        Dataset(subject)

    for subject in Dataset.registry:
        subject.load_data(PATH_TO_DATA, raw=True)
        subject.select_channels(channels=5)
        subject.filter_data(lp_freq=None, hp_freq=1)
        subject.prepare_data(mode=2)
        X, Y = subject.find_best_features(feature_limit=30, 
            scale_data=True,statistic='Anova')

        for idx, cl in enumerate(classifier.registry):
            cl.grid_search_sklearn(X, Y, parameters_list[idx])

        for cl in classifier.registry:
            score = cl.classify(X, Y)
    """
    registry = []
    figures_path = os.path.dirname(os.path.abspath(__file__)) + '/figures'
    os.makedirs(figures_path, exist_ok=True)

    def __init__(self, subject):
        self.name = subject
        self.registry.append(self)
        # TODO is self in self.registry neccesary 
        

    def load_data(self, path_to_data, raw=True):
        """Load subject data from the Kara One dataset.

        Notes
        -----
        By default, the function does not load all the channels, it excludes
        ['EKG', 'EMG', 'Trigger', 'STI 014']. 
        It uses files:
            *.cnt                      raw EEG data
            *-filtered.fif             (optional) filtered data
            all_features_simple.mat    epoch time intervals
            epoch_inds.mat             ordered list of prompts

        Parameters
        ----------
        path_to_data : str
            path to the "p" folder of the database,
            e.g. "/home/USER_NAME/KARA ONE/p/".
        raw : bool
            If true, loads original data (*.cnt). Otherwise loads
            *-filtered.fif, filtered by 
            filter_dat(save_filtered_data=True) in previous runs.
        """
        self.dataPath = path_to_data + self.name
        os.chdir(self.dataPath)
        if raw:
            print("Loading raw data.")
            for f in glob.glob("*.cnt"):
                self.eeg_data = mne.io.read_raw_cnt(f, 'standard_1020', preload=True)
                # self.eeg_data.drop_channels(['CB1', 'CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger', 'STI 014'])
                self.eeg_data.drop_channels(['CB1', 'CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger'])
        else:
            print("Loading filtered data.")
            for f in glob.glob("*-filtered.fif"):
                self.eeg_data = mne.io.read_raw_fif(f, 'standard_1020', preload=True)
        for f in glob.glob("all_features_simple.mat"):
            prompts_to_extract = sio.loadmat(f)
        self.prompts = prompts_to_extract['all_features'][0,0]
        for f in glob.glob("epoch_inds.mat"):
            self.epoch_inds = sio.loadmat(f, variable_names=('clearing_inds', 'thinking_inds'))


    def select_channels(self, channels=5):
        """Choose how many or which channels to use.

        Notes
        -----
        List of available channels in KARA ONE:

        ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
        'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
        'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
        
        Excluded : ['CB1','CB2', 'VEO', 'HEO', 'EKG', 'EMG', 'Trigger', 'STI 014']

        Parameters
        ----------
        channels : int | list
            if int n, it randomly selects n channels.
            If a list of strings, it treats them as chosen channels's names.

        Examples
        --------
        Dataset.select_channels(channels=60)
        Dataset.select_channels(channels=['FP1', 'FPZ', 'FP2', 'AF3'])
        
        """
        # TODO: check if passed channels exist. 
        if type(channels) is int:
            import random
            picked_channels = random.sample(range(len(self.eeg_data.ch_names)), channels)
            picks = [ch for idx, ch in enumerate(self.eeg_data.ch_names) if idx in picked_channels]
            self.eeg_data = self.eeg_data.pick_channels(picks)
        elif type(channels) is list:
            self.eeg_data = self.eeg_data.pick_channels(channels)
        else: 
            raise AttributeError("Incorrect \"channels\" attribute type, should be an int, a list or left empty.")
            # TODO: check if works
        print(self.eeg_data.info['ch_names'])

    def filter_data(self, lp_freq=49, hp_freq=1, save_filtered_data=False, plot=False):
        """Filter subject's EEG data.

        Notes
        -----
        Filter is Butterworth, order is 4.

        Parameters
        ----------
        lp_freq : int
            Frequency of a low-pass filter. Pass a false value (None, 
            False, 0) to disable the filter.
        hp_freq : int
            Frequency of a high-pass filter. Pass a false value (None, 
            False, 0) to disable the filter.
        save_filtered_data : bool 
            Saves filtered data, so it can be loaded later by load_data().
            Data is stored in subject's data folder. 
        plot : bool
            Plots results from before and after a filtration. The results
            are not shown during the runtime. Instead, the are saved, path 
            is stored in self.figures_path, by default /YOUR_SCRIPT/figures.
        """
        print("Filtering data.")
        if plot:
            fig = self.eeg_data.plot_psd(tmax=np.inf, average=False, fmin=0., fmax=130., show=False)
            fig.savefig(os.path.join(self.figures_path, self.name + "_raw_signal.png"))
        if hp_freq:
            for idx, eeg_vector in enumerate(self.eeg_data[:][0]):
                [b, a] = sig.butter(4, hp_freq/self.eeg_data.info['sfreq']/2, btype='highpass')
                self.eeg_data[idx] = sig.filtfilt(b, a, eeg_vector)
        if lp_freq:
            for idx, eeg_vector in enumerate(self.eeg_data[:][0]):
                [b, a] = sig.butter(4, lp_freq/self.eeg_data.info['sfreq']/2, btype='lowpass')
                self.eeg_data[idx] = sig.filtfilt(b, a, eeg_vector)
        print("Filtering done.")
        if plot:
            fig = self.eeg_data.plot_psd(tmax=np.inf, average=False, fmin=0., fmax=130., show=False)
            fig.savefig(os.path.join(self.figures_path, self.name + "_filtered_signal.png"))
        if save_filtered_data:
            self.eeg_data.save(self.name + "-filtered.fif", overwrite=True)
            print("Filtered data saved as " + self.name + "-filtered.fif")


    def ica(self, ica_type='fast'):
        """Exclude components using Independent Component Analysis.

        Notes
        -----
        Requires a manual input concerning components to be excluded.
        Adequate, interactive plots are provided. Results are saved.

        Parameters
        ----------
        ica_type : {'fast', 'extensive', 'saved'}
            'fast'          fast ICA, but with reduced accuracy.
            'extensive'     slow, but more accurate (extended-infomax)
            'saved'         load previously computed.

        Examples
        --------
        Dataset.ica(ica_type('extensive'))
        Dataset.ica(ica_type('saved'))
        """
        if ica_type == 'fast':
            print("Computing fast ICA.")
            ica = mne.preprocessing.ICA(random_state=1)
        elif ica_type == 'saved':
            print("Loading previously computed ICA.")
            ica = mne.preprocessing.read_ica(self.name + "-ica.fif")
        elif ica_type == 'extensive':
            print("Computing extensive ICA.")
            ica = mne.preprocessing.ICA(method='extended-infomax', random_state=1)
        else: 
            raise AttributeError("Incorrect \"ica_type\" attribute value.")
            # TODO: check if works
        ica.fit(self.eeg_data)
        ica.plot_components(inst=self.eeg_data)
        print("Write the components you want to exclude from the data. For instance \"14 28 66\"")
        ica.exclude = [int(x) for x in input().split()]
        ica.apply(self.eeg_data)
        if ica_type != "saved":
            ica.save(self.name + "-ica.fif")


    def prepare_data(self, prompts_list=None, scale_data=True):
        """Prepare labels and calculate features.

        Parameters
        ----------
        prompts_list : dict
            a dictionary, containing chosen prompts as keys 
            and their categories as their respective values. 
            Take a look at the SilentSpeechClassifier.expVariants
            module or example.py for working instances.
        scale_data : bool
            scales features values with 'Sklearn.StandardScaler()'.
            Some ML algorithms demand scaled data to work properly.

        Returns
        -------
        X :ndarray
            2d array of signal features with 'float' type.
        Y : ndarray
            1D array of labels with 'str' type.
        """
       
        def _calculate_features(condition_inds, prompts_list):
            print("Calculating features.")
            offset = int(self.eeg_data.info["sfreq"]/2)
            X = []
            for i, prompt in enumerate(self.prompts["prompts"][0]):
                if prompt[0] in prompts_list:
                    start = self.epoch_inds[condition_inds][0][i][0][0] + offset
                    end = self.epoch_inds[condition_inds][0][i][0][1]
                    channel_set = []
                    for idx, ch in  enumerate(self.eeg_data.ch_names):  
                        epoch = self.eeg_data[idx][0][0][start:end]
                        channel_set.extend(fast_feat_array(epoch, ch))
                    X.append(channel_set)
            return X

        t0 = time()
        Y =[]
        if type(prompts_list) is dict:
            print('Making custom categories.')
            for prompt in self.prompts["prompts"][0]:
                if prompt[0] in prompts_list:
                    Y.append(prompts_list[prompt[0]])
            X = _calculate_features("thinking_inds", prompts_list.keys())
        else:
            print('Rest vs active.')
            Y = self.prompts["prompts"][0]
            X = _calculate_features("clearing_inds", Y)
            X.extend(_calculate_features("thinking_inds", Y))
            Y = np.hstack([np.repeat('rest', len(X)/2), np.repeat('active', len(X)/2)])
        
        print("Features calculated.\nDone in %0.3fs" % (time() - t0))
        self.X = np.asarray(X)
        self.Y = np.hstack(Y) 

        if scale_data:
            print("Scaling data.")
            self.X['feature_value'] = StandardScaler().fit_transform(self.X['feature_value'])

        return self.X['feature_value'], self.Y


    def find_best_features(self, feature_limit=30):
        """Select n best features.

        Notes
        -----
        Reduces dimensionality and redundancy of features. 
        The implementation is based on Anova.

        Parameters
        ----------
        feature_limit : int
            Number of features to leave. 

        Returns
        -------
        X :ndarray
            2d array of signal features with 'float' type.
        Y : ndarray
            1D array of labels with 'str' type.
        """ 
        X = self.X
        Y = self.Y

        print("Calculating ANOVA.")
        selector = SelectKBest(score_func=f_classif, k=feature_limit)
        
        selector.fit(X['feature_value'], Y)
        print(selector.get_support([True]))

        chosen = []
        for idx in selector.get_support([True]):
            chosen.append([selector.scores_[idx], selector.pvalues_[idx], X[0,idx]['channel'], X[0,idx]['feature_name']])
        
        chosen.sort(key=lambda s: s[1])
        for chsn in chosen:
            print("F= %0.3f\tp = %0.3f\t channel = %s\t fname = %s" % (chsn[0], chsn[1], chsn[2], chsn[3]))

        trans_chosen = np.transpose(chosen)
        for chosen, text in (
            (trans_chosen[2], 'Scored by channels: '), 
            (trans_chosen[3], 'Scored by features: ')):
            unique, counts = np.unique(chosen, return_counts=True)
            sorted_counts = sorted(dict(zip(unique, counts)).items(), reverse=True, key=lambda s: s[1])
            print(text ,sorted_counts)
        
        print("ANOVA calculated, ", len(X[0])-feature_limit, "features removed,", feature_limit, " features left.")
        X = selector.transform(X)

        return X['feature_value'], Y


# TODO NN compatibility 
class Classifier:
    """Prepare ML models and classify data.

    Notes
    -----
    The class provides methods for parameters optimisation
    and data classification. Former one utilise exaustive search,
    the latter inputed classifiers in repeated stratified kfold model.

    Algorithm's parameters are not relevant if grid_search_sklearn() is
    to be used, adequate parameters's ranges should be inputed instead.

    Attributes
    ---------
    registry : list
        list of class instances.
    name : str
        name of classifier for logging puproses.
    algorithm : object
        classifier object.

    Methods
    -------
    grid_search_sklearn(self, X, Y, parameters)
        
    classify(self, X, Y, crval_splits=6, crval_repeats=10)

    """
    registry = []

    def __init__(self, name, algorithm):
        self.registry.append(self)
        self.name = name
        self.algorithm = algorithm
    

    def grid_search_sklearn(self, X, Y, parameters):
        """Optimise classifier parameters using exaustive search.

        Parameters
        ----------
        X, Y : array_like
            data for classifier in Sklearn-compatible format. 
        parameters : dict
            Dictionery of parameters for Sklearn.GridSearchCV.
        """
        print('-'*80)
        print("Performing grid search for ", self.name, " algorithm...")
        grid_search = GridSearchCV(self.algorithm, parameters, n_jobs=-2, error_score=0, verbose=0)
        t0 = time()
        grid_search.fit(X, Y)
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best score: %0.3f" % grid_search.best_score_)
        best_parameters = grid_search.best_estimator_.get_params()
        self.algorithm.set_params(**best_parameters)
        print("Best parameters for ", self.name, ":\n", best_parameters)


    def classify(self, X, Y, crval_splits=6, crval_repeats=10):
        """Classify data.

        Notes
        -----
        Repeated stratified K-fold is a cross-validation model, 
        which repeats splitting of the data with a different
        randomization in each iteration. 

        Parameters
        ----------
        X, Y : array_like
            data for classifier in Sklearn-compatible format. 
        crval_splits : int
            Number of splits for cross-validation.
        crval_repeats : int
            Number of repeats for classification
        Returns
        -------
        Accuracy, F1 : list
            Accuracy and F scores from each pass, not averaged. 
        """
        Accuracy = []
        F1 = []
        CFM = []
        rsk = RepeatedStratifiedKFold(n_splits=crval_splits, n_repeats=crval_repeats)
        t0 = time()
        for train, test in rsk.split(X, Y):
            self.algorithm.fit(X[train], Y[train])
            predicted = self.algorithm.predict(X[test])
            Accuracy.append(accuracy_score(Y[test], predicted) * 100)
            F1.append(f1_score(Y[test], predicted, average='macro') * 100)
            CFM.append(confusion_matrix(Y[test], predicted))
        print('-'*40+'\n%s\n'  % self.name + '-'*40)
        print("Parameters: ", self.algorithm.get_params())
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
        print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
        print("\nConfusion Matrix:\n", np.sum(CFM, axis = 0),'\nNumber of instances: ', np.sum(CFM))
        print("done in %0.3fs" % (time() - t0))

        return Accuracy, F1



if __name__ == '__main__':

    pass