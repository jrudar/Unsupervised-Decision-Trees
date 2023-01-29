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

# Import relevant libraries
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats import subsample_counts
from skbio.stats.composition import multiplicative_replacement, closure, clr

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import resample, shuffle
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

import shap as sh

from umap import UMAP

# Function for rarefaction
# https://stackoverflow.com/questions/15507993/quickly-rarefy-a-matrix-in-numpy-python
def rarefaction(M, y1, D, seed=0):
    prng = RandomState(seed)  # reproducible results

    n_occur = M.sum(axis=1)
    rem = np.where(n_occur < depth, False, True)
    M_ss = M[rem]
    n_occur = n_occur[rem]
    nvar = M.shape[1]  # number of variables

    Mrarefied = np.empty_like(M_ss)
    for i in range(M_ss.shape[0]):  # for each sample
        p = M_ss[i] / float(n_occur[i])  # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied[i] = np.bincount(choice, minlength=nvar)

    return Mrarefied, y1[rem]


# Function for creating random data for use in unsupervised learning
def addcl2(X, y):

    X_perm = np.copy(X, "C")
    for col in range(X_perm.shape[0]):
        X_perm[:, col] = np.random.choice(
            X_perm[:, col], replace=False, size=X_perm.shape[0]
        )

    y_new = ["Original" for _ in range(X.shape[0])]
    y_new.extend(["Randomized" for _ in range(X.shape[0])])
    y_new = np.asarray(y_new)

    X_new = np.vstack((X, X_perm))

    return X_new, y_new


# Create positive and negative controls
if __name__ == "__main__":

    # Read in taxa data
    taxa_tab = pd.read_csv("Baxter/rdp.out.tmp", delimiter="\t", header=None).values

    # Keep all ASVs assigned to Bacteria and Archaea, remove Cyanobacteria and Chloroplasts
    idx = np.where(
        ((taxa_tab[:, 2] == "Bacteria") | (taxa_tab[:, 2] == "Archaea")), True, False
    )
    taxa_tab = taxa_tab[idx]
    idx = np.where(taxa_tab[:, 5] != "Cyanobacteria/Chloroplast", True, False)
    taxa_tab = taxa_tab[idx]
    X_selected = set([x[0] for x in taxa_tab])
    taxa_tab_ss = {x[0]: x for x in taxa_tab}

    # Read in ASV table
    X = pd.read_csv("Baxter/ESV.table", index_col=0, sep="\t")
    X_col = [entry.split("_")[0] for entry in X.columns.values]
    X_features = list(set(X.index.values).intersection(X_selected))
    X_index = [s_name.split("_")[0] for s_name in X.columns.values]
    X_signal = X.transpose()[X_features]
    X_signal.index = X_index

    # Get names of high confidence features
    n_list = [4, 7, 10, 13, 16, 19]
    X_name = []
    cluster_name = []
    for row in taxa_tab:
        for entry in X_features:
            if row[0] == entry:
                if float(row[n_list[-1]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-1] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-1] - 2], entry))
                    break

                elif float(row[n_list[-2]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-2] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-2] - 2], entry))
                    break

                elif float(row[n_list[-3]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-3] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-3] - 2], entry))
                    break

                elif float(row[n_list[-4]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-4] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-4] - 2], entry))
                    break

                elif float(row[n_list[-5]]) > 0.8:
                    X_name.append("%s (%s)" % (row[n_list[-5] - 2], entry))
                    cluster_name.append("%s-%s" % (row[n_list[-5] - 2], entry))
                    break

                else:
                    X_name.append("%s" % entry)
                    cluster_name.append("Unclassified-%s" % entry)
                    break

    # Import OTU table and metadata
    meta = pd.read_csv("Baxter/SraRunTable.txt")
    meta = meta[["Run", "Bases", "BioSample", "fit_result", "Diagnosis"]]

    # Only keep biosamples which were sequenced the highest
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

    # Remove any data with NaN
    nan_idx = [i for i, x in enumerate(meta["Diagnosis"].values) if type(x) != float]

    X_signal = X_signal.iloc[nan_idx]
    meta = meta.iloc[nan_idx]

    # Append the FIT data to the ASV table
    X_signal_fit = np.copy(X_signal, "C")
    X_signal_fit = pd.DataFrame(
        X_signal_fit, index=X_signal.index, columns=X_signal.columns
    )
    X_signal_fit["FIT"] = meta["fit_result"].values

    # Simplify Classes
    fixed_diag = []
    for i, entry in enumerate(meta["Diagnosis"].values):
        if "adenoma" in entry.lower():
            fixed_diag.append("Adenoma")

        elif "normal" in entry.lower():
            fixed_diag.append("Normal")

        elif "cancer" in entry.lower():
            fixed_diag.append("Cancer")

    meta["Simple_Diag"] = fixed_diag

    ad_v_no = np.where(meta["Simple_Diag"] != "Cancer", True, False)
    no_v_ca = np.where(meta["Simple_Diag"] != "Adenoma", True, False)
    no_v_le = [True for _ in meta["Simple_Diag"].values]

    # Feature names
    cluster_names = X_features

    def get_result_permanova(X, y, n_rep=999):
        pmanova = permanova(X, y, permutations=n_rep)

        pseudo_f, pval = pmanova.values[4:6]
        R2 = 1 - 1 / (
            1
            + pmanova.values[4]
            * pmanova.values[4]
            / (pmanova.values[2] - pmanova.values[3] - 1)
        )

        return pseudo_f, pval, R2

    def get_perm(X, y, metric, transform_type, comp_type, pcoa_trf, n_neighbors=15):

        if pcoa_trf == 0:
            D = DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32))

        elif pcoa_trf == 1:
            D = pcoa(
                DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32)),
                number_of_dimensions=2,
            ).samples.values

            D = DistanceMatrix(
                pairwise_distances(D, metric="euclidean").astype(np.float32)
            )

        elif pcoa_trf == 2:
            D = UMAP(
                n_components=2, min_dist=0.001, metric=metric, n_neighbors=n_neighbors
            ).fit_transform(X_trn_pa)

            D = DistanceMatrix(
                pairwise_distances(D, metric="euclidean").astype(np.float32)
            )

        elif pcoa_trf == 3:
            D = DistanceMatrix(X)

        per_result = get_result_permanova(D, y)

        return (
            transform_type,
            comp_type,
            metric,
            per_result[0],
            per_result[1],
            per_result[2],
        )

    dataset_types = ["Without_Fit", "With_Fit"]

    datasets = [X_signal, X_signal_fit]

    comp_types = ["Normal_v_Lesion", "Normal_v_Cancer", "Adenoma_v_Cancer"]

    comps_disc = ["Lesion", "Cancer", "Adenoma"]

    # This section of the code creates figures that plots BACC and ROCAUC scores
    get_stats = False
    if get_stats:
        from statannotations.Annotator import Annotator
        from itertools import combinations

        # Plot Graphs - BACC and ROCAUC
        df_perm = pd.read_csv("Baxter_Gen_Perf_Normal_v_Cancer.csv")

        comparison = []
        model_name = []
        scores = []
        metric = []
        FIT = []
        for row in df_perm.values:
            comparison.append(row[1])
            comparison.append(row[1])
            comparison.append(row[1])
            comparison.append(row[1])

            model_name.append(row[2])
            model_name.append(row[2])
            model_name.append(row[2])
            model_name.append(row[2])

            scores.extend(row[3:])
            metric.extend(["BACC", "BACC", "ROC-AUC", "ROC-AUC"])
            FIT.extend(["No", "Yes", "No", "Yes"])

        df_new = pd.DataFrame(
            [comparison, model_name, scores, metric, FIT],
            index=["Comparision", "Model", "Scores", "Metric", "FIT"],
        ).transpose()

        df_fit_bacc_lm = df_new[
            np.where(
                (df_new["FIT"] == "Yes")
                & (df_new["Metric"] == "BACC")
                & (df_new["Model"] == "LANDMark Classifier"),
                True,
                False,
            )
        ]
        np.mean(df_fit_bacc_lm["Scores"])

        df_fit_bacc_lm = df_new[
            np.where(
                (df_new["FIT"] == "Yes")
                & (df_new["Metric"] == "ROC-AUC")
                & (df_new["Model"] == "LANDMark Classifier"),
                True,
                False,
            )
        ]
        np.mean(df_fit_bacc_lm["Scores"])

        order = [
            ("Random", "Yes"),
            ("Extra Trees Classifier", "Yes"),
            ("Random Forest", "Yes"),
            ("LANDMark Classifier", "Yes"),
            ("TreeOrdination", "Yes"),
        ]
        pairs = list(combinations(order, 2))

        fig, ax = plt.subplots(nrows=1, ncols=2)

        # Balanced Accuracy Graph
        loc = np.where(df_new["Metric"] == "BACC", True, False)
        sns.boxplot(
            x="Model", y="Scores", hue="FIT", data=df_new[loc], dodge=True, ax=ax[0]
        )

        annotator = Annotator(
            ax[0], pairs[0:-1], data=df_new[loc], x="Model", y="Scores", hue="FIT"
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            comparisons_correction="fdr_bh",
            loc="outside",
            correction_format="replace",
        )
        annotator.apply_and_annotate()
        ax[0].get_legend().remove()

        # Balanced Accuracy Graph
        loc = np.where(df_new["Metric"] == "ROC-AUC", True, False)
        sns.boxplot(
            x="Model", y="Scores", hue="FIT", data=df_new[loc], dodge=True, ax=ax[1]
        )

        annotator = Annotator(
            ax[1], pairs[0:-1], data=df_new[loc], x="Model", y="Scores", hue="FIT"
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            comparisons_correction="fdr_bh",
            loc="outside",
            correction_format="replace",
        )
        annotator.apply_and_annotate()

        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=8)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90, fontsize=8)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")

        plt.tight_layout()
        plt.savefig("Finalfig.svg")

        plt.close()

    normal_v_cancer = False
    if normal_v_cancer:
        for comp_idx, comp in enumerate([no_v_ca]):

            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

            counter = 1
            for train_index, test_index in splitter.split(
                X_signal_fit.values, meta["Simple_Diag"].values
            ):
                if counter == 1:
                    final_dict = {i: [] for i in range(X_signal_fit.values.shape[0])}

                print(counter)
                counter += 1

                X_train = X_signal_fit.values[train_index]
                X_test = X_signal_fit.values[test_index]

                y_train_idx = np.where(
                    meta["Simple_Diag"].values[train_index] != "Adenoma", True, False
                )
                y_test_idx = np.where(
                    meta["Simple_Diag"].values[test_index] != "Adenoma", True, False
                )

                X_train = X_train[y_train_idx]
                X_test = X_test[y_test_idx]

                y_train = meta["Simple_Diag"].values[train_index][y_train_idx]
                y_test = meta["Simple_Diag"].values[test_index][y_test_idx]

                # Retain ASVs present in at least 5% of samples
                X_sum = np.where(X_train[:, 0:-1] > 0, 1, 0)
                X_sum = np.sum(X_sum, axis=0)
                X_sum = X_sum / X_train.shape[0]
                removed_ASVs = np.where(X_sum >= 0.05, True, False)

                X_ss_train = X_train[:, 0:-1]
                X_ss_train = X_ss_train[:, removed_ASVs]
                X_train = np.hstack((X_ss_train, X_train[:, -1].reshape(-1, 1)))

                X_ss_test = X_test[:, 0:-1]
                X_ss_test = X_ss_test[:, removed_ASVs]
                X_test = np.hstack((X_ss_test, X_test[:, -1].reshape(-1, 1)))

                # Comment out the .fit() and model_results to just run the transformation - If the fit() and model results are used the TreeOrdination model will be run.
                model = TreeOrdination(
                    max_samples_tree=100,
                    metric="hamming",
                    feature_names=np.asarray([str(i) for i in range(X_train.shape[1])]),
                    unsup_n_estim=160,
                    n_iter_unsup=5,
                    n_jobs=64,
                    n_neighbors=8,
                    clr_trf=True,
                    exclude_col=[True, -1],
                )  # .fit(X_train, y_train)
                # model_results = model.predict_proba(X_test)

                X_train_clr = model.scale_clr(X_train)
                X_test_clr = model.scale_clr(X_test)
                del model

                # Uncomment lines below to run LANDMark analysis
                model = LANDMarkClassifier(
                    n_estimators=160, use_nnet=False, n_jobs=32, max_samples_tree=100
                ).fit(X_train_clr, y_train)
                model_results = model.predict_proba(X_test_clr)

                cancer = model_results[:, 0]

                # Save probability scores
                for i, entry in enumerate(test_index[y_test_idx]):
                    final_dict[entry].append(cancer[i])

            # Average probability scores for each sample and save to a CSV file
            cancer = dict()
            for key, value in final_dict.items():
                cancer[key] = np.mean(value)

            fit_scores = X_signal_fit.values[:, -1]

            final_call = []
            cancer_y = []
            fit_x = []
            true_val = []
            fit_pos_neg = []
            y_sel = y_train
            for key, avg_prob in cancer.items():

                final_call.append(meta["Simple_Diag"].values[key])
                cancer_y.append(avg_prob)
                fit_x.append(fit_scores[key])

                if fit_scores[key] >= 100:
                    fit_pos_neg.append("Positive FIT")

                else:
                    fit_pos_neg.append("Negative FIT")

            final_df = pd.DataFrame(
                [fit_x, cancer_y, final_call, fit_pos_neg],
                index=[
                    "FIT Score",
                    "ASV MMT Probability",
                    "Tissue Type",
                    "FIT Prediction",
                ],
            ).transpose()

            final_df.to_csv("Final_DF_cancer_to.csv")

    normal_v_cancer_stat = False
    if normal_v_cancer_stat:

        final_df = pd.read_csv("Final_DF_cancer_to.csv")

        can_norm = np.where(final_df["Tissue Type"] != "Adenoma", True, False)

        # The optimal threshold will maximize the balanced accuracy score for lesions
        thresholds = np.arange(0, 1, 0.01)
        prediction = "Cancer"
        best_threshold = -1
        best_bacc = -1
        tissue_types = final_df["Tissue Type"].values[can_norm]
        scores = final_df["ASV MMT Probability"].values[can_norm]
        for threshold in thresholds:
            predicted = np.where(scores >= threshold, "Cancer", "Normal")

            s = balanced_accuracy_score(tissue_types, predicted)

            if s >= best_bacc:
                best_bacc = s
                best_threshold = threshold

        print(best_threshold)

        # Bootstrap for MMT
        truth = tissue_types

        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(scores, truth, stratify=truth)

            predicted = np.where(scores_resamp >= best_threshold, "Cancer", "Normal")

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Cancer" and truth_resamp[i] == "Cancer":
                    tp += 1

                elif prediction == "Cancer" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Cancer":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu = np.mean(sens_b)
        sens_b_conf = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu = np.mean(spec_b)
        spec_b_conf = np.percentile(spec_b, [2.5, 97.5])

        print("Cancer (MMT)", sens_b_mu, sens_b_conf, spec_b_mu, spec_b_conf)

        # Bootstrap for FIT (Cancer)
        truth = tissue_types

        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(
                final_df["FIT Score"].values[can_norm], truth, stratify=truth
            )

            predicted = np.where(scores_resamp >= 100, "Cancer", "Normal")

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Cancer" and truth_resamp[i] == "Cancer":
                    tp += 1

                elif prediction == "Cancer" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Cancer":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu_fit = np.mean(sens_b)
        sens_b_conf_fit = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu_fit = np.mean(spec_b)
        spec_b_conf_fit = np.percentile(spec_b, [2.5, 97.5])

        print(
            "Cancer (FIT)",
            sens_b_mu_fit,
            sens_b_conf_fit,
            spec_b_mu_fit,
            spec_b_conf_fit,
        )

        # Bootstrap for FIT (Adenoma)
        adenoma_loc = np.where(final_df["Tissue Type"].values != "Cancer", True, False)

        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(
                final_df["FIT Score"].values[adenoma_loc],
                final_df["Tissue Type"].values[adenoma_loc],
                stratify=final_df["Tissue Type"].values[adenoma_loc],
            )

            predicted = np.where(scores_resamp >= 100, "Adenoma", "Normal")

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Adenoma" and truth_resamp[i] == "Adenoma":
                    tp += 1

                elif prediction == "Adenoma" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Adenoma":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu_fit = np.mean(sens_b)
        sens_b_conf_fit = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu_fit = np.mean(spec_b)
        spec_b_conf_fit = np.percentile(spec_b, [2.5, 97.5])

        print(
            "Adenoma (FIT)",
            sens_b_mu_fit,
            sens_b_conf_fit,
            spec_b_mu_fit,
            spec_b_conf_fit,
        )

        g = sns.catplot(
            x="FIT Prediction",
            y="FIT Score",
            hue="Tissue Type",
            col="Tissue Type",
            data=final_df,
        )

        x2, y2 = [0, 1], [100, 100]
        g.axes[0, 0].plot(x2, y2, marker="_", color="k")
        g.axes[0, 0].grid(False)
        g.axes[0, 1].plot(x2, y2, marker="_", color="k")
        g.axes[0, 1].grid(False)
        g.axes[0, 2].plot(x2, y2, marker="_", color="k")
        g.axes[0, 2].grid(False)

        g = sns.catplot(
            x="FIT Prediction",
            y="ASV MMT Probability",
            hue="Tissue Type",
            col="Tissue Type",
            data=final_df,
        )

        x2, y2 = [0, 1], [best_threshold, best_threshold]
        g.axes[0, 0].plot(x2, y2, marker="_", color="k")
        g.axes[0, 0].grid(False)
        g.axes[0, 1].plot(x2, y2, marker="_", color="k")
        g.axes[0, 1].grid(False)
        g.axes[0, 2].plot(x2, y2, marker="_", color="k")
        g.axes[0, 2].grid(False)

        plt.tight_layout()
        plt.savefig("MMT_Result_Cancer_LANDMark.svg")
        plt.close()

    normal_v_lesion = False
    if normal_v_lesion:
        for comp_idx, comp in enumerate([no_v_le]):

            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

            counter = 1
            for train_index, test_index in splitter.split(
                X_signal_fit.values, meta["Simple_Diag"].values
            ):
                if counter == 1:
                    final_dict = {i: [] for i in range(X_signal_fit.values.shape[0])}

                print(counter)
                counter += 1

                X_train = X_signal_fit.values[train_index]
                X_test = X_signal_fit.values[test_index]

                y_train = np.where(
                    meta["Simple_Diag"].values[train_index] != "Normal",
                    "Lesion",
                    "Normal",
                )
                y_test = np.where(
                    meta["Simple_Diag"].values[test_index] != "Normal",
                    "Lesion",
                    "Normal",
                )

                # Retain ASVs present in at least 5% of samples
                X_sum = np.where(X_train[:, 0:-1] > 0, 1, 0)
                X_sum = np.sum(X_sum, axis=0)
                X_sum = X_sum / X_train.shape[0]
                removed_ASVs = np.where(X_sum >= 0.05, True, False)

                X_ss_train = X_train[:, 0:-1]
                X_ss_train = X_ss_train[:, removed_ASVs]
                X_train = np.hstack((X_ss_train, X_train[:, -1].reshape(-1, 1)))

                X_ss_test = X_test[:, 0:-1]
                X_ss_test = X_ss_test[:, removed_ASVs]
                X_test = np.hstack((X_ss_test, X_test[:, -1].reshape(-1, 1)))

                # Comment out the .fit() and model_results to just run the transformation - If the fit() and model results are used the TreeOrdination model will be run.
                model = TreeOrdination(
                    max_samples_tree=100,
                    metric="hamming",
                    feature_names=np.asarray([str(i) for i in range(X_train.shape[1])]),
                    unsup_n_estim=160,
                    n_iter_unsup=5,
                    n_jobs=64,
                    n_neighbors=8,
                    clr_trf=True,
                    exclude_col=[True, -1],
                )  # .fit(X_train, y_train)
                # model_results = model.predict_proba(X_test)
                X_train_clr = model.scale_clr(X_train)
                X_test_clr = model.scale_clr(X_test)
                del model

                # Uncomment the results below to run LANDMark
                model = LANDMarkClassifier(
                    max_samples_tree=100, n_estimators=160, use_nnet=False, n_jobs=32
                ).fit(X_train_clr, y_train)
                model_results = model.predict_proba(X_test_clr)

                cancer = model_results[:, 0]

                for i, entry in enumerate(test_index):
                    final_dict[entry].append(cancer[i])

            cancer = dict()
            for key, value in final_dict.items():
                cancer[key] = np.mean(value)

            fit_scores = X_signal_fit.values[:, -1]

            final_call = []
            cancer_y = []
            fit_x = []
            true_val = []
            fit_pos_neg = []
            y_sel = y_train
            for key, avg_prob in cancer.items():

                final_call.append(meta["Simple_Diag"].values[key])
                cancer_y.append(avg_prob)
                fit_x.append(fit_scores[key])

                if fit_scores[key] >= 100:
                    fit_pos_neg.append("Positive FIT")

                else:
                    fit_pos_neg.append("Negative FIT")

            final_df = pd.DataFrame(
                [fit_x, cancer_y, final_call, fit_pos_neg],
                index=[
                    "FIT Score",
                    "ASV MMT Probability",
                    "Tissue Type",
                    "FIT Prediction",
                ],
            ).transpose()

            final_df.to_csv("Final_DF_lesion_to.csv")

    normal_v_lesion_stat = False
    if normal_v_lesion_stat:

        final_df = pd.read_csv("Final_DF_lesion_to.csv")

        # The optimal threshold will maximize the balanced accuracy score for lesions
        thresholds = np.arange(0, 1, 0.01)
        prediction = "Cancer"
        prediction_2 = "Adenoma"
        best_threshold = -1
        best_bacc = -1
        tissue_types = np.where(
            final_df["Tissue Type"].values == prediction,
            "Lesion",
            final_df["Tissue Type"].values,
        )
        tissue_types = np.where(tissue_types == prediction_2, "Lesion", tissue_types)
        scores = final_df["ASV MMT Probability"].values
        for threshold in thresholds:
            predicted = np.where(scores >= threshold, "Lesion", "Normal")

            s = balanced_accuracy_score(tissue_types, predicted)
            print(threshold, s)
            if s >= best_bacc:
                best_bacc = s
                best_threshold = threshold

        print(best_threshold)

        # For Cancer
        # Bootstrap for MMT
        truth = final_df["Tissue Type"].values

        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(scores, truth, stratify=truth)

            predicted = np.where(scores_resamp >= best_threshold, "Cancer", "Normal")

            cancer_v_normal = np.where(truth_resamp != "Adenoma", True, False)
            predicted = predicted[cancer_v_normal]
            truth_resamp = truth_resamp[cancer_v_normal]

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Cancer" and truth_resamp[i] == "Cancer":
                    tp += 1

                elif prediction == "Cancer" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Cancer":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu = np.mean(sens_b)
        sens_b_conf = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu = np.mean(spec_b)
        spec_b_conf = np.percentile(spec_b, [2.5, 97.5])

        print("Cancer", sens_b_mu, sens_b_conf, spec_b_mu, spec_b_conf)

        # For Adenoma
        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(scores, truth, stratify=truth)

            predicted = np.where(scores_resamp >= best_threshold, "Adenoma", "Normal")

            cancer_v_normal = np.where(truth_resamp != "Cancer", True, False)
            predicted = predicted[cancer_v_normal]
            truth_resamp = truth_resamp[cancer_v_normal]

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Adenoma" and truth_resamp[i] == "Adenoma":
                    tp += 1

                elif prediction == "Adenoma" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Adenoma":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu = np.mean(sens_b)
        sens_b_conf = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu = np.mean(spec_b)
        spec_b_conf = np.percentile(spec_b, [2.5, 97.5])

        print("Adenoma", sens_b_mu, sens_b_conf, spec_b_mu, spec_b_conf)

        # For Lesion
        sens_b = []
        spec_b = []

        for _ in range(2000):
            scores_resamp, truth_resamp = resample(scores, truth, stratify=truth)

            predicted = np.where(scores_resamp >= best_threshold, "Lesion", "Normal")

            predicted = predicted
            truth_resamp = truth_resamp
            truth_resamp = np.where(truth_resamp != "Normal", "Lesion", "Normal")

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i, prediction in enumerate(predicted):
                if prediction == "Lesion" and truth_resamp[i] == "Lesion":
                    tp += 1

                elif prediction == "Lesion" and truth_resamp[i] == "Normal":
                    fp += 1

                elif prediction == "Normal" and truth_resamp[i] == "Normal":
                    tn += 1

                elif prediction == "Normal" and truth_resamp[i] == "Lesion":
                    fn += 1

            sens_ca = tp / (tp + fn)
            spec_ca = tn / (tn + fp)

            sens_b.append(sens_ca)
            spec_b.append(spec_ca)

        sens_b_mu = np.mean(sens_b)
        sens_b_conf = np.percentile(sens_b, [2.5, 97.5])

        spec_b_mu = np.mean(spec_b)
        spec_b_conf = np.percentile(spec_b, [2.5, 97.5])

        print("Lesion", sens_b_mu, sens_b_conf, spec_b_mu, spec_b_conf)

        g = sns.catplot(
            x="FIT Prediction",
            y="ASV MMT Probability",
            hue="Tissue Type",
            col="Tissue Type",
            data=final_df,
        )

        x2, y2 = [0, 1], [best_threshold, best_threshold]
        g.axes[0, 0].plot(x2, y2, marker="_", color="k")
        g.axes[0, 0].grid(False)
        g.axes[0, 1].plot(x2, y2, marker="_", color="k")
        g.axes[0, 1].grid(False)
        g.axes[0, 2].plot(x2, y2, marker="_", color="k")
        g.axes[0, 2].grid(False)

        plt.tight_layout()
        plt.savefig("MMT_Result_Lesion_LANDMark.svg")
        plt.close()

    get_fi_scores = False
    if get_fi_scores:
        # Feature names
        feature_names = np.asarray(cluster_name)

        # Create a test-train split (index only)
        y_training = np.where(
            meta["Simple_Diag"].values != "Normal", "Lesion", "Normal"
        )

        X_train, X_test, y_train, y_test, y_tr, y_te = train_test_split(
            X_signal_fit.values,
            y_training,
            meta["Simple_Diag"].values,
            train_size=0.8,
            random_state=0,
            stratify=meta["Simple_Diag"].values,
        )

        # Retain ASVs present in at least 5% of samples
        X_sum = np.where(X_train[:, 0:-1] > 0, 1, 0)
        X_sum = np.sum(X_sum, axis=0)
        X_sum = X_sum / X_train.shape[0]
        removed_ASVs = np.where(X_sum >= 0.05, True, False)

        X_ss_train = X_train[:, 0:-1]
        X_ss_train = X_ss_train[:, removed_ASVs]
        X_train = np.hstack((X_ss_train, X_train[:, -1].reshape(-1, 1)))

        X_ss_test = X_test[:, 0:-1]
        X_ss_test = X_ss_test[:, removed_ASVs]
        X_test = np.hstack((X_ss_test, X_test[:, -1].reshape(-1, 1)))

        feature_names = feature_names[removed_ASVs]

        # Train the model
        model = TreeOrdination(
            metric="hamming",
            feature_names=feature_names,
            unsup_n_estim=160,
            n_iter_unsup=5,
            n_jobs=10,
            n_neighbors=8,
            clr_trf=True,
            max_samples_tree=100,
            exclude_col=[True, -1],
        ).fit(X_train, y_train)

        best_threshold = 0.66  # Identified previously

        # Get probs for lesion for training and testing data
        train_prob = model.predict_proba(X_train)[:, 0]
        test_prob = model.predict_proba(X_test)[:, 0]

        # Divide training and testing samples into FIT Positive and FIT Negative groups
        fit_pos_test_ca = np.where(
            (X_test[:, -1] >= 100) & (test_prob >= best_threshold), True, False
        )  # Get all FIT positive samples predicted to be lesions
        fit_neg_test_ca = np.where(
            (X_test[:, -1] < 100) & (test_prob >= best_threshold), True, False
        )  # Get all FIT negative samples predicted to be lesions

        fit_pos_test_no = np.where(
            (X_test[:, -1] >= 100) & (test_prob < best_threshold) & (y_te == "Normal"),
            True,
            False,
        )  # GET ALL FIT positve and negative normal samples
        fit_neg_test_no = np.where(
            (X_test[:, -1] < 100) & (test_prob < best_threshold) & (y_te == "Normal"),
            True,
            False,
        )

        # Plot data
        test_emb_true = model.emb_transform(X_test)
        test_emb = model.approx_emb(X_test)
        X_1 = test_emb[fit_pos_test_ca]
        y_label_1 = ["%s" % x for x in y_te[fit_pos_test_ca]]
        style_1 = ["FIT Positive" for _ in X_1]

        X_2 = test_emb[fit_neg_test_ca]
        y_label_2 = ["%s" % x for x in y_te[fit_neg_test_ca]]
        style_2 = ["FIT Negative" for _ in X_2]

        X_3 = test_emb[fit_pos_test_no]
        y_label_3 = ["%s" % x for x in y_te[fit_pos_test_no]]
        style_3 = ["FIT Positive" for _ in X_3]

        X_4 = test_emb[fit_neg_test_no]
        y_label_4 = ["%s" % x for x in y_te[fit_neg_test_no]]
        style_4 = ["FIT Negative" for _ in X_4]

        X_comb = np.vstack((X_1, X_2, X_3, X_4))
        y_label = np.hstack((y_label_1, y_label_2, y_label_3, y_label_4))
        style = np.hstack((style_1, style_2, style_3, style_4))

        sns.scatterplot(x=X_comb[:, 0], y=X_comb[:, 1], hue=y_label, style=style)

        pc1 = model.R_PCA.explained_variance_ratio_[0] * 100
        pc2 = model.R_PCA.explained_variance_ratio_[1] * 100
        plt.xlabel("PCA 1 (%.3f Percent)" % pc1)
        plt.ylabel("PCA 2 (%.3f Percent)" % pc2)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("TreeOrd_Proj_Lesion_MMT.svg")
        plt.close()

        # Get PerMANOVA of projection
        perm_res = get_perm(
            X=X_comb,
            y=np.where(
                (np.asarray(y_label) == "Cancer") | (np.asarray(y_label) == "Adenoma"),
                "Lesion",
                "Normal",
            ),
            metric="euclidean",
            transform_type="TreeOrd",
            comp_type="TreeOrd",
            pcoa_trf=0,
            n_neighbors=15,
        )
        print(perm_res)

        # Get SHAP Scores
        X_test_clr = model.scale_clr(X_test)

        X_subset = np.vstack(
            (
                X_test_clr[fit_pos_test_ca],
                X_test_clr[fit_neg_test_ca],
                X_test_clr[fit_pos_test_no],
                X_test_clr[fit_neg_test_no],
            )
        )

        predictive_model = model.l_model
        mse_test = mean_squared_error(test_emb, test_emb_true)
        print(mse_test)

        E = sh.Explainer(
            predictive_model, feature_names=np.hstack((feature_names, ["FIT Score"]))
        )
        shap_test = E(X_test_clr)

        ca_loc = np.where(
            (X_test[:, -1] < 100) & (test_prob >= best_threshold) & (y_te == "Cancer"),
            True,
            False,
        )

        do_shap_plots = True
        if do_shap_plots:
            # Cancer FIT Positive PCA 1 and 2
            ca_loc = np.where(
                (X_test[:, -1] >= 100)
                & (test_prob >= best_threshold)
                & (y_te == "Cancer"),
                True,
                False,
            )
            sh.plots.bar(shap_test[ca_loc, :, 0], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ca_fit_pos.svg")
            plt.close()

            sh.plots.bar(shap_test[ca_loc, :, 1], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ca_fit_pos.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 0], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ca_fit_pos_example.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 1], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ca_fit_pos_example.svg")
            plt.close()

            # Cancer FIT Negative PCA 1 and 2
            ca_loc = np.where(
                (X_test[:, -1] < 100)
                & (test_prob >= best_threshold)
                & (y_te == "Cancer"),
                True,
                False,
            )
            sh.plots.bar(shap_test[ca_loc, :, 0], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ca_fit_neg.svg")
            plt.close()

            sh.plots.bar(shap_test[ca_loc, :, 1], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ca_fit_neg.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 0], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ca_fit_neg_example.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 1], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc2_test_ca_fit_neg_example.svg")
            plt.close()

            # Adenoma FIT Positive PCA 1 and 2
            ca_loc = np.where(
                (X_test[:, -1] >= 100)
                & (test_prob >= best_threshold)
                & (y_te == "Adenoma"),
                True,
                False,
            )
            sh.plots.bar(shap_test[ca_loc, :, 0], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ad_fit_pos.svg")
            plt.close()

            sh.plots.bar(shap_test[ca_loc, :, 1], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ad_fit_pos.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 0], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ad_fit_pos_example.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 1], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ad_fit_pos_example.svg")
            plt.close()

            # Adenoma FIT Negative PCA 1 and 2
            ca_loc = np.where(
                (X_test[:, -1] < 100)
                & (test_prob >= best_threshold)
                & (y_te == "Adenoma"),
                True,
                False,
            )
            sh.plots.bar(shap_test[ca_loc, :, 0], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ad_fit_neg.svg")
            plt.close()

            sh.plots.bar(shap_test[ca_loc, :, 1], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ad_fit_neg.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 0], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_ad_fit_neg_example.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 1], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_ad_fit_neg_example.svg")
            plt.close()

            # Normal FIT Negative PCA 1 and 2
            ca_loc = np.where(
                (X_test[:, -1] < 100)
                & (test_prob < best_threshold)
                & (y_te == "Normal"),
                True,
                False,
            )
            sh.plots.bar(shap_test[ca_loc, :, 0], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_no_fit_neg.svg")
            plt.close()

            sh.plots.bar(shap_test[ca_loc, :, 1], show=False, max_display=10)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_no_fit_neg.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 0], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc0_test_no_fit_neg_example.svg")
            plt.close()

            sh.plots.waterfall(shap_test[ca_loc][0, :, 1], show=False)
            plt.tight_layout()
            plt.savefig("shap_pc1_test_no_fit_neg_example.svg")
            plt.close()

        # Separate the adenomas from cancers
        lesions = np.where(
            (test_prob >= best_threshold) & ((y_te == "Cancer") | (y_te == "Adenoma")),
            True,
            False,
        )
        lesion_type = [x for i, x in enumerate(y_te) if lesions[i] == True]

        # Plot a graph that shows the differneces between lesions
        sh.plots.bar(shap_test[lesions, :, 0].cohorts(lesion_type).abs.mean(0))

        # Calculate PerMANOVA for these samples
        perm_res = get_perm(
            X=test_emb[lesions],
            y=lesion_type,
            metric="euclidean",
            transform_type="TreeOrd",
            comp_type="TreeOrd",
            pcoa_trf=0,
            n_neighbors=15,
        )
        print(perm_res)
