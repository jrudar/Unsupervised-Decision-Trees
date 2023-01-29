#!/usr/bin/env python
# coding: utf-8

from __future__ import division
# utils
import pandas as pd
import numpy as np
from collections import Counter
# blocks
from scipy.stats import norm
from numpy.random import poisson, lognormal
from skbio.stats.composition import closure
from scipy.special import kl_div
from scipy.stats import entropy
# minimize model perams
from sklearn.metrics import mean_squared_error, balanced_accuracy_score, roc_auc_score
from scipy.optimize import minimize

# Import NumPy and Pandas
import numpy as np
import pandas as pd

#Import relevant libraries
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats.composition import multiplicative_replacement, closure, clr

from scipy.spatial import procrustes

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances, balanced_accuracy_score
from sklearn.utils import resample, shuffle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA

import umap as um

import seaborn as sns

import matplotlib.pyplot as plt

from LANDMark import LANDMarkClassifier
from TreeOrdination import TreeOrdination

from deicode.preprocessing import rclr
from deicode.matrix_completion import MatrixCompletion

from numpy.random import RandomState

from umap import UMAP

from numpy import __version__
print("numpy", __version__)

from pandas import __version__
print("pandas", __version__)

from skbio import __version__
print("skbio", __version__)

from scipy import __version__
print("scipy", __version__)

from sklearn import __version__
print("sklearn", __version__)

from umap import __version__
print("umap", __version__)

from seaborn import __version__
print("seaborn", __version__)

from matplotlib import __version__
print("matplotlib", __version__)

from deicode import __version__
print("deicode", __version__)

from shap import __version__
print("shap", __version__)

#Function for rarefaction
#https://stackoverflow.com/questions/15507993/quickly-rarefy-a-matrix-in-numpy-python
def rarefaction(M, y1, D, seed=0):
    prng = RandomState(seed) # reproducible results

    n_occur = M.sum(axis = 1)
    rem = np.where(n_occur < depth, False, True)
    M_ss = M[rem]
    n_occur = n_occur[rem]
    nvar = M.shape[1] # number of variables

    Mrarefied = np.empty_like(M_ss)
    for i in range(M_ss.shape[0]): # for each sample
        p = M_ss[i] / float(n_occur[i]) # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied[i] = np.bincount(choice, minlength=nvar)

    return Mrarefied, y1[rem]

#Function for creating random data for use in unsupervised learning
def addcl2(X, y):
    
    X_perm = np.copy(X, "C")
    for col in range(X_perm.shape[0]):
        X_perm[:, col] = np.random.choice(X_perm[:, col], replace = False, size = X_perm.shape[0])
        
    y_new = ["Original" for _ in range(X.shape[0])]
    y_new.extend(["Randomized" for _ in range(X.shape[0])])
    y_new = np.asarray(y_new)
    
    X_new = np.vstack((X, X_perm))
            
    return X_new, y_new

#Create positive and negative controls
if __name__ == "__main__":
    
    #Read in taxa data
    taxa_tab = pd.read_csv("Baxter/rdp.out.tmp", delimiter = "\t", header = None).values

    #Keep all ASVs assigned to Bacteria and Archaea, remove Cyanobacteria and Chloroplasts
    idx = np.where(((taxa_tab[:, 2] == "Bacteria") | (taxa_tab[:, 2] == "Archaea")), True, False)
    taxa_tab = taxa_tab[idx]
    idx = np.where(taxa_tab[:, 5] != "Cyanobacteria/Chloroplast", True, False)
    taxa_tab = taxa_tab[idx]
    X_selected = set([x[0] for x in taxa_tab])
    taxa_tab_ss = {x[0]: x for x in taxa_tab}

    #Read in ASV table
    X = pd.read_csv("Baxter/ESV.table", index_col = 0, sep = "\t")
    X_col = [entry.split("_")[0] for entry in X.columns.values]
    X_features = list(set(X.index.values).intersection(X_selected))
    X_index = [s_name.split("_")[0] for s_name in X.columns.values]
    X_signal = X.transpose()[X_features]
    X_signal.index = X_index

    #Import OTU table and metadata
    meta = pd.read_csv("Baxter/SraRunTable.txt")
    meta = meta[["Run", "Bases", "BioSample", 'fit_result', 'Diagnosis']]

    #Only keep biosamples which were sequenced the highest
    unique_biosample_idx = dict()
    for i, (biosample, n_bases) in enumerate(meta[["BioSample", "Bases"]].values):
        if biosample not in unique_biosample_idx:
            unique_biosample_idx[biosample] = [i, n_bases]
        
        elif biosample in unique_biosample_idx:
            if unique_biosample_idx[biosample][1] >= n_bases:
                unique_biosample_idx[biosample] = [i, n_bases]

    bio_idx = np.asarray([n_bases[0] for idx, n_bases in unique_biosample_idx.items()])

    meta = meta.iloc[bio_idx]
    X_signal = X_signal.loc[meta["Run"]]

    #Remove any data with NaN
    nan_idx = [i for i, x in enumerate(meta["Diagnosis"].values) if type(x) != float]

    X_signal = X_signal.iloc[nan_idx]
    meta = meta.iloc[nan_idx]

    #Append the FIT data to the ASV table
    X_signal_fit = np.copy(X_signal, "C")
    X_signal_fit = pd.DataFrame(X_signal_fit, index = X_signal.index, columns = X_signal.columns)
    X_signal_fit["FIT"] = meta["fit_result"].values

    #Simplify Classes
    fixed_diag = []
    for i, entry in enumerate(meta["Diagnosis"].values):
        if "adenoma" in entry.lower():
            fixed_diag.append("Adenoma")
        
        elif "normal" in entry.lower():
            fixed_diag.append("Normal")

        elif "cancer" in entry.lower():
            fixed_diag.append("Cancer")
        
    meta["Simple_Diag"] = fixed_diag

    #Get locations of each class in the dataset and metadata
    ad_v_no = np.where(meta["Simple_Diag"] != "Cancer", True, False)
    no_v_ca = np.where(meta["Simple_Diag"] != "Adenoma", True, False)
    no_v_le = np.asarray([True for _ in X_signal.values]) 

    #Feature names
    cluster_names = X_features
    
    """
        #Functions for calculating PerMANOVA and Generalization Performance
        def get_result_model(model, X_tr, y_tr, X_te, y_te):
            #Fit the model
            model = model.fit(X_tr, y_tr)

            #Predict class labels
            pred_tests = model.predict(X_te)

            #Return BACC
            ba_tests = balanced_accuracy_score(y_te, pred_tests)

            return ba_tests

        def get_result_permanova(X, y, n_rep = 999):
            pmanova = permanova(X, y, permutations = n_rep)

            pseudo_f, pval = pmanova.values[4:6]
            R2 = 1 - 1 / (1 + pmanova.values[4] * pmanova.values[4] / (pmanova.values[2] - pmanova.values[3] - 1))

            return pseudo_f, pval, R2

        def get_perm(X, y, metric, transform_type, comp_type, pcoa_trf, n_neighbors = 15):

            if pcoa_trf == 0:
                D = DistanceMatrix(pairwise_distances(X, metric = metric).astype(np.float32))

            elif pcoa_trf == 1:
                D = pcoa(DistanceMatrix(pairwise_distances(X, 
                                                            metric = metric).astype(np.float32)),
                            number_of_dimensions = 2).samples.values

                D = DistanceMatrix(pairwise_distances(D, metric = "euclidean").astype(np.float32))

            elif pcoa_trf == 2:
                D = UMAP(n_components = 2, min_dist = 0.001, metric = metric, n_neighbors = n_neighbors).fit_transform(X)

                D = DistanceMatrix(pairwise_distances(D, metric = "euclidean").astype(np.float32))

            elif pcoa_trf == 3:
                D = DistanceMatrix(X)

            per_result = get_result_permanova(D, y)

            return transform_type, comp_type, metric, per_result[0], per_result[1], per_result[2]

        def get_classifer(X, y, X_te, y_te, metric, model, transform_type, comp_type, model_type, pcoa_trf, n_neighbors = 15):

            if pcoa_trf == 0:
                result = get_result_model(model, X, y, X_te, y_te)

            elif pcoa_trf == 1:
                X_trf = UMAP(n_components = 2, min_dist = 0.001, metric = metric, n_neighbors = n_neighbors).fit(X)
                X_tr = X_trf.transform(X)
                X_test_proj = X_trf.transform(X_te)

                result = get_result_model(model, X_tr, y, X_test_proj, y_te)

            elif pcoa_trf == 2:
                X_tr = closure(X)
                X_test_proj = closure(X_te)

                result = get_result_model(model, X_tr, y, X_test_proj, y_te)

            elif pcoa_trf == 3:
                X_tr = clr(multiplicative_replacement(closure(X)))
                X_test_proj = clr(multiplicative_replacement(closure(X_te)))

                result = get_result_model(model, X_tr, y, X_test_proj, y_te)

            return transform_type, comp_type, model_type, metric, result
        """
    dataset_types = ["Without_Fit", "With_Fit"]
    datasets = [X_signal, X_signal_fit]
    comp_types = ["Normal_v_Cancer", "Normal_v_Lesion", "Adenoma_v_Cancer"]

    #Cross-validation - Loop through signal and random data   
    for comp_type, comp in enumerate([no_v_ca, no_v_le, ad_v_no]):
        
        #List of balanced accuracy scores (test and validation) and PerMANOVA data for each iteratiion
        BAS_data = []
        PER_data = []
        scores = []

        #5x5 Stratified Cross-Validation - Positive Control  
        splitter = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 5, random_state = 0)

        counter = 1

        data_reduced_nofit = datasets[0][comp]
        data_reduced_wifit = datasets[1][comp]
        meta_reduced = meta[comp]

        for train, tests in splitter.split(data_reduced_nofit, meta_reduced["Simple_Diag"].values): 

            print("Iteration Number:", counter)
            counter += 1

            X_train_nofit = data_reduced_nofit.values[train]
            X_tests_nofit = data_reduced_nofit.values[tests]

            X_train_wifit = data_reduced_wifit.values[train]
            X_tests_wifit = data_reduced_wifit.values[tests]

            y_train = meta_reduced["Simple_Diag"].values[train]
            y_tests = meta_reduced["Simple_Diag"].values[tests]

            if comp_type == 1:
                y_train = np.where(y_train != "Normal", "Lesion", "Normal")
                y_tests = np.where(y_tests != "Normal", "Lesion", "Normal")

            #Retain ASVs present in at least 5% of samples
            X_sum = np.where(X_train_nofit > 0, 1, 0)
            X_sum = np.sum(X_sum, axis = 0)
            X_sum = X_sum / X_train_nofit.shape[0]
            removed_ASVs = np.where(X_sum >= 0.05, True, False)

            feature_names = np.asarray(cluster_names)[removed_ASVs]

            X_train_nofit = X_train_nofit[:, removed_ASVs]
            X_tests_nofit = X_tests_nofit[:, removed_ASVs]

            X_train_wifit = X_train_wifit[:, np.hstack((removed_ASVs, [True]))]
            X_tests_wifit = X_tests_wifit[:, np.hstack((removed_ASVs, [True]))]

            #####################################################
            X_train_clr = clr(multiplicative_replacement(closure(X_train_nofit)))
            X_tests_clr = clr(multiplicative_replacement(closure(X_tests_nofit)))

            X_train_clr_wfit = np.hstack((X_train_clr, X_train_wifit[:, -1].reshape(-1,1)))
            X_tests_clr_wfit = np.hstack((X_tests_clr, X_tests_wifit[:, -1].reshape(-1,1)))

            clf_rnd = DummyClassifier(strategy = "stratified").fit(X_train_clr , y_train)
            pred = clf_rnd.predict(X_tests_clr)
            s_1 = balanced_accuracy_score(y_tests, pred)
            roc_auc_1 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr)[:, 1])
            print(comp_types[comp_type], "Random: BACC = %.3f ROC-AUC = %.3f" %(s_1, roc_auc_1))

            clf_rnd = DummyClassifier(strategy = "stratified").fit(X_train_clr_wfit , y_train)
            pred = clf_rnd.predict(X_tests_clr_wfit)
            s_2 = balanced_accuracy_score(y_tests, pred)
            roc_auc_2 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr_wfit)[:, 1])
            print(comp_types[comp_type], "Random: BACC = %.3f ROC-AUC = %.3f" %(s_2, roc_auc_2))
            scores.append((comp_types[comp_type], "Random", s_1, s_2, roc_auc_1, roc_auc_2))
               
            clf_rnd = RandomForestClassifier(160).fit(X_train_clr , y_train)
            pred = clf_rnd.predict(X_tests_clr)
            s_1 = balanced_accuracy_score(y_tests, pred)
            roc_auc_1 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr)[:, 1])
            print(comp_types[comp_type], "RF: BACC = %.3f ROC-AUC = %.3f" %(s_1, roc_auc_1))

            clf_rnd = RandomForestClassifier(160).fit(X_train_clr_wfit , y_train)
            pred = clf_rnd.predict(X_tests_clr_wfit)
            s_2 = balanced_accuracy_score(y_tests, pred)
            roc_auc_2 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr_wfit)[:, 1])
            print(comp_types[comp_type], "RF: BACC = %.3f ROC-AUC = %.3f" %(s_2, roc_auc_2))
            scores.append((comp_types[comp_type], "Random Forest", s_1, s_2, roc_auc_1, roc_auc_2))
    
            clf_rnd = ExtraTreesClassifier(160).fit(X_train_clr , y_train)
            pred = clf_rnd.predict(X_tests_clr)
            s_1 = balanced_accuracy_score(y_tests, pred)
            roc_auc_1 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr)[:, 1])
            print(comp_types[comp_type], "ET: BACC = %.3f ROC-AUC = %.3f" %(s_1, roc_auc_1))

            clf_rnd = ExtraTreesClassifier(160).fit(X_train_clr_wfit , y_train)
            pred = clf_rnd.predict(X_tests_clr_wfit)
            s_2 = balanced_accuracy_score(y_tests, pred)
            roc_auc_2 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr_wfit)[:, 1])
            print(comp_types[comp_type], "ET: BACC = %.3f ROC-AUC = %.3f" %(s_2, roc_auc_2))
            scores.append((comp_types[comp_type], "Extra Trees Classifier", s_1, s_2, roc_auc_1, roc_auc_2))
    
            clf_rnd =LANDMarkClassifier(160, 
                                        n_jobs = 32,
                                        max_samples_tree = 100,
                                        use_nnet = False).fit(X_train_clr , y_train)
            pred = clf_rnd.predict(X_tests_clr)
            s_1 = balanced_accuracy_score(y_tests, pred)
            roc_auc_1 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr)[:, 1])
            print(comp_types[comp_type], "LM: BACC = %.3f ROC-AUC = %.3f" %(s_1, roc_auc_1))

            clf_rnd = LANDMarkClassifier(160, 
                                        n_jobs = 32,
                                        max_samples_tree = 100,
                                        use_nnet = False).fit(X_train_clr_wfit , y_train)
            pred = clf_rnd.predict(X_tests_clr_wfit)
            s_2 = balanced_accuracy_score(y_tests, pred)
            roc_auc_2 = roc_auc_score(y_tests, clf_rnd.predict_proba(X_tests_clr_wfit)[:, 1])
            print(comp_types[comp_type], "LM: BACC = %.3f ROC-AUC = %.3f" %(s_2, roc_auc_2))
            scores.append((comp_types[comp_type], "LANDMark Classifier", s_1, s_2, roc_auc_1, roc_auc_2))

            clf_tr = TreeOrdination(metric = "hamming", feature_names = feature_names, unsup_n_estim = 160, n_iter_unsup = 5, n_jobs = 32, max_samples_tree = 100, n_neighbors = 8, clr_trf = True).fit(X_train_nofit, y_train)
            ps = clf_tr.predict_proba(X_tests_nofit)
            pred = clf_tr.predict(X_tests_nofit)
            s_1 = balanced_accuracy_score(y_tests, pred)
            roc_auc_1 = roc_auc_score(y_tests, ps[:, 1])
            print(comp_types[comp_type], "TreeOrdination: BACC = %.3f ROC-AUC = %.3f" %(s_1, roc_auc_1))
    
            clf_tr = TreeOrdination(metric = "hamming", feature_names = feature_names, unsup_n_estim = 160, n_iter_unsup = 5, n_jobs = 32, max_samples_tree = 100, n_neighbors = 8, clr_trf = True, exclude_col = [True, -1]).fit(X_train_wifit, y_train)
            ps = clf_tr.predict_proba(X_tests_wifit)
            pred = clf_tr.predict(X_tests_wifit)
            s = balanced_accuracy_score(y_tests, pred)
            roc_auc = roc_auc_score(y_tests, ps[:, 1])
            print(comp_types[comp_type], "TreeOrdination: BACC = %.3f ROC-AUC = %.3f" %(s_2, roc_auc_2))
            scores.append((comp_types[comp_type], "TreeOrdination", s_1, s_2, roc_auc_1, roc_auc_2))

        scores = pd.DataFrame(scores, columns = ["Comparison", "Model", "No Fit (BACC)", "FIT (BACC)", "No Fit (ROC-AUC)", "FIT (ROC-AUC)"])
        scores.to_csv("Baxter_Gen_Perf_%s.csv" %comp_types[comp_type])
