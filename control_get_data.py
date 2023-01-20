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
from sklearn.metrics import mean_squared_error, balanced_accuracy_score
from scipy.optimize import minimize

# Import relevant libraries
from skbio.stats.distance import permanova, DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats import subsample_counts
from skbio.stats.composition import multiplicative_replacement, closure, clr

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import ExtraTreesClassifier

import seaborn as sns

import matplotlib.pyplot as plt

# For printing graphs
from itertools import combinations
from statannotations.Annotator import Annotator

from LANDMark import LANDMarkClassifier
from TreeOrdination import TreeOrdination

# For RCLR and RPCA
from deicode.preprocessing import rclr
from deicode.matrix_completion import MatrixCompletion

from numpy.random import RandomState

from umap import UMAP

# Function for rarefaction
# https://stackoverflow.com/questions/15507993/quickly-rarefy-a-matrix-in-numpy-python
def rarefaction(M, y1, M2, y2, D, seed=0):
    prng = RandomState(seed)  # reproducible results

    n_occur = M.sum(axis=1)
    rem = np.where(n_occur < depth, False, True)
    M_ss = M[rem]
    n_occur = n_occur[rem]
    nvar = M.shape[1]  # number of variables

    # Do training data
    Mrarefied = np.empty_like(M_ss)
    for i in range(M_ss.shape[0]):  # for each sample
        p = M_ss[i] / float(n_occur[i])  # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied[i] = np.bincount(choice, minlength=nvar)

    # Do testing data - Rarefy to same depth as training data
    n_occur2 = M2.sum(axis=1)
    rem2 = np.where(n_occur2 < depth, False, True)
    M_ss2 = M2[rem2]
    n_occur2 = n_occur2[rem2]
    nvar = M2.shape[1]  # number of variables

    Mrarefied2 = np.empty_like(M_ss2)
    for i in range(M_ss2.shape[0]):  # for each sample
        p = M_ss2[i] / float(n_occur2[i])  # relative frequency / probability
        choice = prng.choice(nvar, D, p=p)
        Mrarefied2[i] = np.bincount(choice, minlength=nvar)

    return Mrarefied, y1[rem], Mrarefied2, y2[rem2]


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


"""
    The following five functions are from:
    https://github.com/cameronmartino/deicode-benchmarking/blob/master/simulations/scripts/simulations.py
    Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A, Knight R, et al. A Novel Sparse Compositional Technique Reveals Microbial Perturbations. mSystems. 2019 Feb;4(1). 
"""
# Set random state
rand = np.random.RandomState(42)


def Homoscedastic(X_noise, intensity):
    """uniform normally dist. noise"""
    X_noise = np.array(X_noise)
    err = intensity * np.ones_like(X_noise.copy())
    X_noise = rand.normal(X_noise.copy(), err)

    return X_noise


def Heteroscedastic(X_noise, intensity):
    """non-uniform normally dist. noise"""
    err = intensity * np.ones_like(X_noise)
    i = rand.randint(0, err.shape[0], 5000)
    j = rand.randint(0, err.shape[1], 5000)
    err[i, j] = intensity
    X_noise = abs(rand.normal(X_noise, err))

    return X_noise


def Subsample(X_noise, spar, num_samples):
    """yij ~ PLN( lambda_{ij}, /phi )"""
    # subsample
    mu = spar * closure(X_noise.T).T
    X_noise = np.vstack(
        [poisson(lognormal(np.log(mu[:, i]), 1)) for i in range(num_samples)]
    ).T
    # add sparsity

    return X_noise


def block_diagonal_gaus(ncols, nrows, nblocks, overlap=0, minval=0, maxval=1.0):
    """
    Generate block diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    ncol : int
        Number of columns

    nrows : int
        Number of rows

    nblocks : int
        Number of blocks, mucst be greater than one

    overlap : int
        The Number of overlapping columns (Default = 0)

    minval : int
        The min value output of the table (Default = 0)

    maxval : int
        The max value output of the table (Default = 1)


    Returns
    -------
    np.array
        Table with a block diagonal where the rows represent samples
        and the columns represent features.  The values within the blocks
        are gaussian distributed between 0 and 1.
    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    if nblocks <= 1:
        raise ValueError("`nblocks` needs to be greater than 1.")
    mat = np.zeros((nrows, ncols))
    gradient = np.linspace(0, 10, nrows)
    mu = np.linspace(0, 10, ncols)
    sigma = 1
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma) for i in range(len(mu))]
    mat = np.vstack(xs).T

    block_cols = ncols // nblocks
    block_rows = nrows // nblocks
    for b in range(nblocks - 1):

        gradient = np.linspace(5, 5, block_rows)  # samples (bock_rows)
        # features (block_cols+overlap)
        mu = np.linspace(0, 10, block_cols + overlap)
        sigma = 2.0
        xs = [norm.pdf(gradient, loc=mu[i], scale=sigma) for i in range(len(mu))]

        B = np.vstack(xs).T * maxval
        lower_row = block_rows * b
        upper_row = min(block_rows * (b + 1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols * (b + 1), ncols)

        if b == 0:
            mat[lower_row:upper_row, lower_col : int(upper_col + overlap)] = B
        else:
            ov_tmp = int(overlap / 2)
            if (B.shape) == (
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp + 1),
                ].shape
            ):
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp + 1),
                ] = B
            elif (B.shape) == (
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp),
                ].shape
            ):
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp),
                ] = B
            elif (B.shape) == (
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp - 1),
                ].shape
            ):
                mat[
                    lower_row:upper_row,
                    int(lower_col - ov_tmp) : int(upper_col + ov_tmp - 1),
                ] = B

    upper_col = int(upper_col - overlap)
    # Make last block fill in the remainder
    gradient = np.linspace(5, 5, nrows - upper_row)
    mu = np.linspace(0, 10, ncols - upper_col)
    sigma = 4
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma) for i in range(len(mu))]
    B = np.vstack(xs).T * maxval

    mat[upper_row:, upper_col:] = B

    return mat


def build_block_model(
    rank, hoced, hsced, spar, C_, num_samples, num_features, overlap=0, mapping_on=True
):
    """
    Generates hetero and homo scedastic noise on base truth block diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    rank : int
        Number of blocks


    hoced : int
        Amount of homoscedastic noise

    hsced : int
        Amount of heteroscedastic noise

    inten : int
        Intensity of the noise

    spar : int
        Level of sparsity

    C_ : int
        Intensity of real values

    num_features : int
        Number of rows

    num_samples : int
        Number of columns

    overlap : int
        The Number of overlapping columns (Default = 0)

    mapping_on : bool
        if true will return pandas dataframe mock mapping file by block


    Returns
    -------
    Pandas Dataframes
    Table with a block diagonal where the rows represent samples
    and the columns represent features.  The values within the blocks
    are gaussian.

    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    # make a mock OTU table
    X_true = block_diagonal_gaus(
        num_samples, num_features, rank, overlap, minval=0.01, maxval=C_
    )
    if mapping_on:
        # make a mock mapping data
        mappning_ = pd.DataFrame(
            np.array(
                [
                    ["Cluster %s" % str(x)] * int(num_samples / rank)
                    for x in range(1, rank + 1)
                ]
            ).flatten(),
            columns=["example"],
            index=["sample_" + str(x) for x in range(0, num_samples - 2)],
        )

    X_noise = X_true.copy()
    X_noise = np.array(X_noise)
    # add Homoscedastic noise
    X_noise = Homoscedastic(X_noise, hoced)
    # add Heteroscedastic noise
    X_noise = Heteroscedastic(X_noise, hsced)
    # Induce low-density into the matrix
    X_noise = Subsample(X_noise, spar, num_samples)

    # return the base truth and noisy data
    if mapping_on:
        return X_true, X_noise, mappning_
    else:
        return X_true, X_noise


# Fit model and return balanced accuracy scores
def get_result_model(model, X_tr, y_tr, X_te, y_te):
    # Fit the model
    model = model.fit(X_tr, y_tr)

    # Predict class labels
    pred_tests = model.predict(X_te)

    # Return BACC
    ba_tests = balanced_accuracy_score(y_te, pred_tests)

    return ba_tests


# Return PerMANOVA results
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


# Proper PerMANOVA test
def get_perm(X, y, metric, transform_type, comp_type, pcoa_trf, n_neighbors=15):

    # Calculate distances
    if pcoa_trf == 0:
        D = DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32))

    # Calculate PCoA representation and then PerMANOVA
    elif pcoa_trf == 1:
        D = pcoa(
            DistanceMatrix(pairwise_distances(X, metric=metric).astype(np.float32)),
            number_of_dimensions=2,
        ).samples.values

        D = DistanceMatrix(pairwise_distances(D, metric="euclidean").astype(np.float32))

    # UMAP and then PerMANOVA
    elif pcoa_trf == 2:
        D = UMAP(
            n_components=2, min_dist=0.001, metric=metric, n_neighbors=n_neighbors
        ).fit_transform(X)

        D = DistanceMatrix(pairwise_distances(D, metric="euclidean").astype(np.float32))

    # Precomputed distances
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


# Training of classifier
def get_classifer(
    X,
    y,
    X_te,
    y_te,
    metric,
    model,
    transform_type,
    comp_type,
    model_type,
    pcoa_trf,
    n_neighbors=15,
):

    # Only use the base transform (eg: Presence-Absence)
    if pcoa_trf == 0:
        result = get_result_model(model, X, y, X_te, y_te)

    # Train classifier using UMAP representation
    elif pcoa_trf == 1:
        X_trf = UMAP(
            n_components=2, min_dist=0.001, metric=metric, n_neighbors=n_neighbors
        ).fit(X)
        X_tr = X_trf.transform(X)
        X_test_proj = X_trf.transform(X_te)

        result = get_result_model(model, X_tr, y, X_test_proj, y_te)

    return transform_type, comp_type, model_type, metric, result


# Create positive and negative controls
if __name__ == "__main__":
    depth = 2.5e3
    overlap_ = 0
    rank_ = 2
    # run model with fit variables and new variants
    _, X_signal = build_block_model(
        rank_,
        depth / 60,
        depth / 60,
        depth,
        depth,
        200,
        1000,
        overlap=overlap_,
        mapping_on=False,
    )

    # run model with fit variables and new variants
    _, X_signal = build_block_model(
        rank_,
        depth / 60,
        depth / 60,
        depth,
        depth,
        200,
        1000,
        overlap=overlap_,
        mapping_on=False,
    )

    X_signal = pd.DataFrame(
        X_signal,
        index=["OTU_" + str(x) for x in range(X_signal.shape[0])],
        columns=["sample_" + str(x) for x in range(X_signal.shape[1])],
    )

    # run model with fit variables and new variants
    X_random = np.random.randint(0, np.mean(X_signal.values) * 2.3, (1000, 200))
    X_random = pd.DataFrame(
        X_random,
        index=["OTU_" + str(x) for x in range(X_random.shape[0])],
        columns=["sample_" + str(x) for x in range(X_random.shape[1])],
    )
    X_random.index = shuffle(X_random).index
    X_random.columns = shuffle(X_random.T).index
    X_random = X_random.T
    X_random.sort_index(inplace=True)
    X_random = X_random.T
    X_random.sort_index(inplace=True)

    # metadata on cluster
    meta = np.array(
        [1] * int(X_signal.shape[1] / 2) + [2] * int(X_signal.shape[1] / 2)
    ).T
    meta = pd.DataFrame(meta, index=X_signal.columns, columns=["group"])

    print("X_random mean %.2f seq/sample" % X_random.sum(axis=0).mean())
    print("X_signal mean %.2f seq/sample" % X_signal.sum(axis=0).mean())

    # Final negative and positive control matrix
    X_random = X_random.transpose()
    X_signal = X_signal.transpose()

    # Feature names
    cluster_names = np.asarray(["ASV_%s" % str(i) for i in range(1000)])

    experiment = ["Positive_Control", "Negative_Control"]

    #Runs the PerMANOVA and Classification Experiments for Positive and Negative Control Data
    run_exp = True
    if run_exp == True:
        # Cross-validation - Loop through signal and random data
        for dataset_type, dataset in enumerate([X_signal, X_random]):
            # List of balanced accuracy scores (test and validation) and PerMANOVA data for each iteratiion
            BAS_data = []
            PER_data = []

            # 5x5 Stratified Cross-Validation - Positive Control
            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)

            counter = 1

            for train, tests in splitter.split(dataset.values, meta["group"].values):

                print("Iteration Number:", counter)
                counter += 1

                X_train = dataset.values[train]
                X_tests = dataset.values[tests]

                y_train = meta["group"].values[train]
                y_tests = meta["group"].values[tests]

                # Retain ASVs present in more than 5% of samples
                ASV_as_row = X_train.transpose()
                totals = ASV_as_row.sum(axis=1)
                five_percent = float(X_train.shape[0] * 0.05)
                removed_ASVs = np.where(totals > five_percent, True, False)

                feature_names = cluster_names[removed_ASVs]

                X_train = X_train[:, removed_ASVs]
                X_tests = X_tests[:, removed_ASVs]

                # Rarefy to the 15th percentile
                noccur = np.sum(
                    X_train, axis=1
                )  # number of occurrences for each sample
                depth = int(np.percentile(noccur, float(15.0)))  # sampling depth

                X_train_rare, y_train_rare, X_tests_rare, y_tests_rare = rarefaction(
                    X_train, y_train, X_tests, y_tests, depth, seed=0
                )

                # Get randomized data to build trees - PA, Bray-Curtis
                X_rnd_1, y_rnd_1 = addcl2(X_train_rare, y_train_rare)
                X_rnd_2, y_rnd_2 = addcl2(X_train_rare, y_train_rare)
                X_rnd_3, y_rnd_3 = addcl2(X_train_rare, y_train_rare)
                X_rnd_4, y_rnd_4 = addcl2(X_train_rare, y_train_rare)
                X_rnd_5, y_rnd_5 = addcl2(X_train_rare, y_train_rare)
                X_rnd = np.vstack((X_rnd_1, X_rnd_2, X_rnd_3, X_rnd_4, X_rnd_5))
                y_rnd = np.hstack((y_rnd_1, y_rnd_2, y_rnd_3, y_rnd_4, y_rnd_5))

                # Get randomized data to build trees - CLR and RCLR (PerMANOVA only)
                X_rnd_1_f, y_rnd_1_f = addcl2(X_train, y_train)
                X_rnd_2_f, y_rnd_2_f = addcl2(X_train, y_train)
                X_rnd_3_f, y_rnd_3_f = addcl2(X_train, y_train)
                X_rnd_4_f, y_rnd_4_f = addcl2(X_train, y_train)
                X_rnd_5_f, y_rnd_5_f = addcl2(X_train, y_train)
                X_rnd_f = np.vstack(
                    (X_rnd_1_f, X_rnd_2_f, X_rnd_3_f, X_rnd_4_f, X_rnd_5_f)
                )
                y_rnd_f = np.hstack(
                    (y_rnd_1_f, y_rnd_2_f, y_rnd_3_f, y_rnd_4_f, y_rnd_5_f)
                )

                #####################################################
                """
                Presence-Absence Transformation
                """
                do_pa = True
                print("Presence-Absence")
                if do_pa == True:
                    # Convert training, testing, and validation data to presence-absence
                    X_trn_pa = np.where(X_train_rare > 0, 1, 0)
                    X_tst_pa = np.where(X_tests_rare > 0, 1, 0)
                    X_rnd_pa = np.where(X_rnd > 0, 1, 0)

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "jaccard",
                                "Presence-Absence",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Presence-Absence",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "jaccard",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "jaccard",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Presence-Absence",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Presence-Absence",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Presence-Absence",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                        ).fit(X_trn_pa, y_train_rare)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Presence-Absence",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        p_result = et_unsup.predict(X_tst_pa)
                        bas_test = balanced_accuracy_score(y_tests_rare, p_result)
                        BAS_data.append(
                            (
                                "Presence-Absence",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )
                        print(BAS_data[-1])

                """
                Proportions and Bray-Curtis Transformation
                """
                do_bc = True
                print("Proportions and Bray-Curtis")
                if do_bc == True:
                    # Convert training, testing, and validation data to proportions
                    X_trn_pa = closure(X_train_rare)
                    X_tst_pa = closure(X_tests_rare)
                    X_rnd_pa = closure(X_rnd)

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportions",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportion",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train_rare,
                                "braycurtis",
                                "Proportion",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                ExtraTreesClassifier(160),
                                "Proportions",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "None",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Proportions",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "braycurtis",
                                ExtraTreesClassifier(160),
                                "Bray-Curtis",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train_rare,
                                X_tst_pa,
                                y_tests_rare,
                                "braycurtis",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Bray-Curtis",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Proportions",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train_rare,
                                leaves_tests,
                                y_tests_rare,
                                "hamming",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Proportions",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                            scale=True,
                            n_neighbors=8,
                        ).fit(X_train_rare, y_train_rare)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train_rare,
                                "Learned",
                                "Proportions",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        p_result = et_unsup.predict(X_tst_pa)
                        bas_test = balanced_accuracy_score(y_tests_rare, p_result)
                        BAS_data.append(
                            (
                                "Proportions",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )
                        print(BAS_data[-1])

                """
                CLR Transformation
                """
                do_clr = True
                print("Centered Log-Ratio")
                if do_clr == True:
                    # Convert training, testing, and validation data to presence-absence
                    X_trn_pa = clr(multiplicative_replacement(closure(X_train)))
                    X_tst_pa = clr(multiplicative_replacement(closure(X_tests)))
                    X_rnd_pa = clr(multiplicative_replacement(closure(X_rnd_f)))

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "Original Distances",
                                0,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Centered Log-Ratio",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "None",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "Original Data",
                                "Extra Trees",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "None",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Centered Log-Ratio",
                                "Original Data",
                                "LANDMark",
                                0,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "euclidean",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "UMAP",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                        BAS_data.append(
                            get_classifer(
                                X_trn_pa,
                                y_train,
                                X_tst_pa,
                                y_tests,
                                "euclidean",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Centered Log-Ratio",
                                "UMAP",
                                "LANDMark",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.apply(X_trn_pa)
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        # Step 1: Get leaves for test and train data, Encode Leaves
                        leaves_test = et_unsup.apply(X_tst_pa)
                        leaves_all = np.vstack((leaves, leaves_test))
                        leaves_trf = OneHotEncoder(sparse=False).fit(leaves_all)

                        leaves_train = leaves_trf.transform(leaves)
                        leaves_tests = leaves_trf.transform(leaves_test)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train,
                                leaves_tests,
                                y_tests,
                                "hamming",
                                ExtraTreesClassifier(160),
                                "Centered Log-Ratio",
                                "Extra Trees Embedding",
                                "Extra Trees",
                                1,
                                8,
                            )
                        )
                        print(BAS_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.proximity(X_trn_pa)
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        # Calculate Generalization Performance
                        leaves_train = leaves
                        leaves_tests = et_unsup.proximity(X_tst_pa)

                        BAS_data.append(
                            get_classifer(
                                leaves_train,
                                y_train,
                                leaves_tests,
                                y_tests,
                                "hamming",
                                LANDMarkClassifier(
                                    160, n_jobs=32, max_samples_tree=100, use_nnet=False
                                ),
                                "Centered Log-Ratio",
                                "LANDMark Embedding",
                                "LANDMark",
                                1,
                                8,
                            )
                        )

                        print(BAS_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                            n_neighbors=8,
                            clr_trf=True,
                        ).fit(X_train, y_train)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Centered Log-Ratio",
                                "TreeOrdination",
                                3,
                            )
                        )

                        p_result = et_unsup.predict(X_tests)
                        bas_test = balanced_accuracy_score(y_tests, p_result)
                        BAS_data.append(
                            (
                                "Centered Log-Ratio",
                                "TreeOrdination Embedding",
                                "TreeOrdination",
                                "Learned",
                                bas_test,
                            )
                        )

                        print(BAS_data[-1])
                        print(PER_data[-1])

                """
                RCLR Transformation
                """
                do_rclr = True
                print("Robust Centered Log-Ratio")
                if do_rclr == True:

                    X_prop_train = np.copy(X_train, "C")
                    X_prop_train = rclr(X_prop_train.transpose()).transpose()
                    M = MatrixCompletion(2, max_iterations=1000).fit(X_prop_train)
                    X_trn_pa = M.U
                    D = M.distance

                    X_rnd_pa = np.copy(X_rnd_f, "C")
                    X_rnd_pa = rclr(X_rnd_pa.transpose()).transpose()
                    M = MatrixCompletion(2, max_iterations=1000).fit(X_rnd_pa)
                    X_rnd_pa = M.solution

                    do_raw = True
                    if do_raw:

                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "euclidean",
                                "Robust Centered Log-Ratio",
                                "Original Distances",
                                3,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                X_trn_pa,
                                y_train,
                                "euclidean",
                                "Robus Centered Log-Ratio",
                                "PCoA",
                                1,
                            )
                        )
                        print(PER_data[-1])
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "precomputed",
                                "Robust Centered Log-Ratio",
                                "UMAP",
                                2,
                                8,
                            )
                        )
                        print(PER_data[-1])

                    do_et = True
                    if do_et:
                        # Train Unsupervised ETC model
                        """
                        1) Fit the model on the original and randomized data
                        2) Extract all terminal leaves
                        3) Create the co-occurance matrix (Equation 4)
                        4) Convert to dissimilarity (Equation 5)

                        Note, although we are using all the samples to get the leaves
                        only the training data was used to create the model.
                        """
                        et_unsup = ExtraTreesClassifier(160).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.apply(X_rnd_pa[0 : X_trn_pa.shape[0]])
                        leaves_binary = OneHotEncoder(sparse=False).fit_transform(
                            leaves
                        )
                        S_xi_xj = np.dot(leaves_binary, leaves_binary.T)
                        S_xi_xj = S_xi_xj / 160
                        D_xi_xj = np.sqrt(1 - S_xi_xj).astype(np.float32)

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(n_components=2, min_dist=0.001, metric="precomputed")
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised Extremely Randomized Trees (UMAP)",
                                3,
                                8,
                            )
                        )
                        print(PER_data[-1])

                    do_lm = True
                    if do_lm:
                        et_unsup = LANDMarkClassifier(
                            160, n_jobs=32, max_samples_tree=100, use_nnet=False
                        ).fit(X_rnd_pa, y_rnd_f)
                        leaves = et_unsup.proximity(X_rnd_pa[0 : X_trn_pa.shape[0]])
                        D_xi_xj = pairwise_distances(leaves, metric="hamming").astype(
                            np.float32
                        )

                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised LANDMark",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            pcoa(
                                DistanceMatrix(D_xi_xj), number_of_dimensions=2
                            ).samples.values
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "Unsupervised LANDMark (PCoA)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                        D = pairwise_distances(
                            UMAP(
                                n_components=2,
                                min_dist=0.001,
                                metric="precomputed",
                                n_neighbors=8,
                            )
                            .fit_transform(D_xi_xj)
                            .astype(np.float32)
                        )
                        PER_data.append(
                            get_perm(
                                D,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratioe",
                                "Unsupervised LANDMark (UMAP)",
                                3,
                            )
                        )
                        print(PER_data[-1])

                    do_to = True
                    if do_to:
                        # TreeOrdination distance matrix
                        et_unsup = TreeOrdination(
                            metric="hamming",
                            feature_names=feature_names,
                            unsup_n_estim=160,
                            n_iter_unsup=5,
                            n_jobs=32,
                            max_samples_tree=100,
                            n_neighbors=8,
                            rclr_trf=True,
                        ).fit(X_train, y_train)
                        D_xi_xj = pairwise_distances(
                            et_unsup.R_PCA_emb, metric="euclidean"
                        ).astype(np.float32)
                        PER_data.append(
                            get_perm(
                                D_xi_xj,
                                y_train,
                                "Learned",
                                "Robust Centered Log-Ratio",
                                "TreeOrdination",
                                3,
                            )
                        )
                        print(PER_data[-1])

            BAS_data = pd.DataFrame(
                BAS_data,
                columns=[
                    "Transformation",
                    "Comparision Type",
                    "Model",
                    "Metric",
                    "BACC",
                ],
            )
            BAS_data.to_csv("%s_BACC.csv" % experiment[dataset_type])

            PER_data = pd.DataFrame(
                PER_data,
                columns=[
                    "Transformation",
                    "Comparision Type",
                    "Metric",
                    "Pseudo-F",
                    "p-value",
                    "R-Squared",
                ],
            )
            PER_data.to_csv("%s_PerMANOVA.csv" % experiment[dataset_type])

    #Create Figures
    get_stats_perm = False
    if get_stats_perm:
        # Comment out of of the two below
        df = pd.read_csv("Negative_Control_PerMANOVA.csv")
        # df = pd.read_csv("Negative_Control_PerMANOVA.csv")

        df["Log (Pseudo-F)"] = np.log(df["Pseudo-F"])

        df["Transformation"] = np.where(
            (df["Transformation"] == "Robust Log-Ratio")
            | (df["Transformation"] == "Robust Centered Log-Ratios")
            | (df["Transformation"] == "Robust Centered Log-Ratioe")
            | (df["Transformation"] == "Robus Centered Log-Ratio"),
            "Robust Centered Log-Ratio",
            df["Transformation"],
        )
        df["Transformation"] = np.where(
            df["Transformation"] == "Proportion", "Proportions", df["Transformation"]
        )
        df["Metric"] = [x.capitalize() for x in df["Metric"].values]

        g = sns.catplot(
            data=df,
            x="Comparision Type",
            y="Pseudo-F",
            hue="Metric",
            col="Transformation",
            kind="bar",
            ci=95,
            n_boot=2000,
            dodge=False,
        )

        g.set_xticklabels(rotation=90)

        plt.tight_layout()

        plt.show()

        plt.close()

    get_stats_bacc = False
    if get_stats_bacc:
        # Comment out one of the two below
        # df = pd.read_csv("Positive_Control_BACC.csv")
        df = pd.read_csv("Negative_Control_BACC.csv")

        df["Transformation"] = np.where(
            (df["Transformation"] == "Robust Log-Ratio")
            | (df["Transformation"] == "Robust Centered Log-Ratios")
            | (df["Transformation"] == "Robust Centered Log-Ratioe"),
            "Robust Centered Log-Ratio",
            df["Transformation"],
        )
        df["Transformation"] = np.where(
            df["Transformation"] == "Proportion", "Proportions", df["Transformation"]
        )
        df["Transformation"] = np.where(
            df["Transformation"] == "Bray-Curtis", "Proportions", df["Transformation"]
        )
        df["Metric"] = np.where(df["Metric"] == "hamming", "Learned", df["Metric"])

        df["Metric"] = [x.capitalize() for x in df["Metric"].values]

        df["Model (Metric)"] = [
            "%s (%s)" % (x, df["Metric"].values[i])
            for i, x in enumerate(df["Model"].values)
        ]

        do_annot = True

        # Set Figure and Axes
        fig, ax = plt.subplots(nrows=1, ncols=3)

        # Statistical analysis preparation
        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Jaccard)",
            "LANDMark (Jaccard)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Presence-Absence", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[0],
        )
        ax[0].set_title("Presence-Absence")
        ax[0].get_legend().remove()
        for tick in ax[0].get_xticklabels():
            tick.set_rotation(90)

        if do_annot:
            annotator = Annotator(
                ax[0], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
            )
            annotator.configure(
                test="Wilcoxon",
                text_format="star",
                comparisons_correction="fdr_bh",
                loc="outside",
                correction_format="replace",
            )
            annotator.apply_and_annotate()

        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Braycurtis)",
            "LANDMark (Braycurtis)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Proportions", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[1],
        )
        ax[1].set_title("Proportion")
        ax[1].get_legend().remove()
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)

        if do_annot:
            annotator = Annotator(
                ax[1], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
            )
            annotator.configure(
                test="Wilcoxon",
                text_format="star",
                comparisons_correction="fdr_bh",
                loc="outside",
                correction_format="replace",
            )
            annotator.apply_and_annotate()

        order = [
            "Extra Trees (None)",
            "LANDMark (None)",
            "Extra Trees (Euclidean)",
            "LANDMark (Euclidean)",
            "Extra Trees (Learned)",
            "LANDMark (Learned)",
            "TreeOrdination (Learned)",
        ]
        pairs = list(combinations(order, 2))
        df_ss = np.where(df["Transformation"] == "Centered Log-Ratio", True, False)
        sns.barplot(
            data=df[df_ss],
            x="Model (Metric)",
            y="BACC",
            hue="Comparision Type",
            ci=95,
            n_boot=2000,
            dodge=False,
            ax=ax[2],
        )
        ax[2].set_title("Centered Log-Ratio")
        for tick in ax[2].get_xticklabels():
            tick.set_rotation(90)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="center left")

        if do_annot:
            annotator = Annotator(
                ax[2], pairs, data=df[df_ss], x="Model (Metric)", y="BACC"
            )
            annotator.configure(
                test="Wilcoxon",
                text_format="star",
                comparisons_correction="fdr_bh",
                loc="outside",
                correction_format="replace",
            )
            annotator.apply_and_annotate()
