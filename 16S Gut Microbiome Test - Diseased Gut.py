from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE, RFECV
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np

import pandas as pd

from random import shuffle

import seaborn as sns
from matplotlib import pyplot as plt

import umap as um

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning) 

from LANDMark import LANDMarkClassifier

def addcl1(X, y):

    X_new = np.copy(X, "C")

    X_new = np.random.permutation(X_new.T).T

    y_new = ["Original" for _ in range(X.shape[0])]
    y_new.extend(["Randomized" for _ in range(X.shape[0])])
    
    y_new = np.asarray(y_new)
    X_new = np.vstack((X, X_new))

    return X_new, y_new

#Taxa to filter out
gp_asvs = {"Firmicutes", "Actinobacteria", "Tenericutes"}

taxa_tab = pd.read_csv("Diseased Gut/rdp.out.tmp", delimiter = "\t", header = None).values

idx = np.where(((taxa_tab[:, 5] == "Firmicutes") | (taxa_tab[:, 5] == "Actinobacteria") | (taxa_tab[:, 5] == "Tenericutes")), True, False)
taxa_tab = taxa_tab[idx]
idx = np.where(taxa_tab[:, 5] != "Cyanobacteria/Chloroplast", True, False)
taxa_tab = taxa_tab[idx]
cutoff = np.where(taxa_tab[:, 19] >= 0.8, True, False)
ASVs = taxa_tab[cutoff, 0]

#Read in ASVs
X = pd.read_csv("Diseased Gut/ESV.table", index_col = 0, sep = "\t")
X_col = [entry.split("_")[0] for entry in X.columns.values]
X = X.loc[ASVs]
X = X.values.transpose()

#Convert to presence-absence and identify filter out rare ASVs
X_pa = np.where(X > 0, 1, 0)
X_sum = np.sum(X_pa, axis = 0)
X_feat = np.where(X_sum > 2, True, False)

print(X.shape)

#CLR transformation adn filtering
from skbio.stats.composition import multiplicative_replacement, closure, clr
X_clr = clr(multiplicative_replacement(closure(X[:, X_feat])))
X_pa = X_pa[:, X_feat]
X_orig = np.copy(X[:, X_feat], "C")

print(X_clr.shape)

#Read in metadata
y = pd.read_csv("Diseased Gut/metadata.csv", index_col = 0)
y = y[["Sample Name", "Host_disease", "Timepoint"]]
y_index = y.index.values
y = y.values

#Split time points
y_time_1 = np.where(y[:, 2].astype(int) > 1, True, False)
y_time_2 = np.where(y[:, 2].astype(int) < 2, True, False)

y_index_1 = y_index[y_time_1]
y_index_2 = y_index[y_time_2]

X_pa_t1 = pd.DataFrame(X_pa, index = X_col).loc[y_index_1].values
X_clr_t1 = pd.DataFrame(X_clr, index = X_col).loc[y_index_1].values
X_orig_t1 = pd.DataFrame(X_orig, index = X_col).loc[y_index_1].values
y_t1 = y[y_time_1, 1]
X_pa_t2 = pd.DataFrame(X_pa, index = X_col).loc[y_index_2].values
X_clr_t2 = pd.DataFrame(X_clr, index = X_col).loc[y_index_2].values
X_orig_t2 = pd.DataFrame(X_orig, index = X_col).loc[y_index_2].values
y_t2 = y[y_time_2, 1]

#Calculate dissimilarity matrix and create PCoA graphs
unsupervised_comparison = False
if unsupervised_comparison == True:

    X = X_clr_t1
    y = y_t1
    
    n = np.sqrt(X_clr.shape[1]) / X_clr.shape[1]

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC", 
             "UC-HC"]

    from sklearn.metrics import pairwise_distances, balanced_accuracy_score
    from skbio.stats.distance import permanova, DistanceMatrix
    from skbio.stats.ordination import pcoa
    from sklearn.preprocessing import OneHotEncoder

    for phenotypes in pheno:

        pheno_a, pheno_b = phenotypes.split("-")

        idx = np.where(((y == pheno_a) | (y == pheno_b)), True, False)

        fig, axs = plt.subplots(4, 2)
        fig.set_figheight(11)
        fig.set_figwidth(8.5)

        #Transform and plot using just distance
        X_1 = X[idx]

        #Transform and plot - Jaccard
        D_ja = pairwise_distances(X_1, metric = "euclidean") #jaccard/euclidean

        E_ja = um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D_ja)

        pc_trf = pcoa(pairwise_distances(E_ja, metric = "euclidean").astype(np.float32))
        E_ja = pc_trf.samples.values

        stat = permanova(DistanceMatrix(D_ja.astype(np.float32)), y[idx])
        R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
        stat["R2"] = R2
        print(stat, R2)

        stat.to_csv("Diseased Gut/%s_%s_permanova_distances_Jaccard.csv" %(pheno_a, pheno_b))
        pd.DataFrame(D_ja, index = y[idx], columns = y[idx]).to_csv("Diseased Gut/%s_%s_distance_matrix_PA.csv" %(pheno_a, pheno_b))
   
        stat = stat.values

        title = "Jaccard Distance (UMAP with PCoA)"
        pc1 = str(pc_trf.proportion_explained[0] * 100)[0:5]
        pc2 = str(pc_trf.proportion_explained[1] * 100)[0:5]

        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[0, 0])

        axs[0, 0].set_title(title)
        axs[0, 0].set_xlabel("PC 1 (%s Percent)" %pc1)
        axs[0, 0].set_ylabel("PC 2 (%s Percent)" %pc2)
        axs[0, 0].legend_ = None

        stat = pcoa(D_ja.astype(np.float32))
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        title = "Jaccard Distance (PCoA)"
        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[0, 1])

        axs[0, 1].set_title(title)
        axs[0, 1].set_xlabel("PCoA 1 (%s Percent)" %pc1)
        axs[0, 1].set_ylabel("PCoA 2 (%s Percent)" %pc2)
        axs[0, 1].legend_ = None

        #Classifiy
        S_lm = np.zeros((X[idx].shape[0], X[idx].shape[0]))
        S_et = np.zeros((X[idx].shape[0], X[idx].shape[0]))
        S_rf = np.zeros((X[idx].shape[0], X[idx].shape[0]))
        for i in range(100):
            #Create a random labeled set of data
            print("Iteration:", i+1)

            clr_T = True
            if clr_T == True:
                X_new, y_new = addcl1(X_orig_t1[idx], y[idx])
                X_new = clr(multiplicative_replacement(closure(X_new)))

            else:
                X_new, y_new = addcl1(X[idx], y[idx])

            X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, test_size = 0.2, stratify = y_new)

            model_lm = LANDMarkClassifier(n_jobs = 5, max_features = n).fit(X_tr.astype(np.float), y_tr)
            model_et = ExtraTreesClassifier().fit(X_tr.astype(np.float), y_tr)
            model_rf = RandomForestClassifier().fit(X_tr.astype(np.float), y_tr)

            print(balanced_accuracy_score(y_te, model_lm.predict(X_te)),
                    balanced_accuracy_score(y_te, model_et.predict(X_te)),
                    balanced_accuracy_score(y_te, model_rf.predict(X_te)))

            #LANDMark Proximity
            leaves = model_lm.proximity(X[idx].astype(np.float))
            M = np.dot(leaves, leaves.T)
            M =  M / np.max(M)
            S_lm += M

            #Random Forest/Extra Trees Proximity
            leaves = model_et.apply(X[idx])
            M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
            M = np.dot(M, M.T)
            M = M / M.max()
            S_et += M

            leaves = model_rf.apply(X[idx])
            M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
            M = np.dot(M, M.T)
            M = M / M.max()
            S_rf += M

        S_lm = 1 - (S_lm / S_lm.max())
        D_lm = np.sqrt(S_lm).astype(np.float32)
        pd.DataFrame(D_lm, index = y[idx], columns = y[idx]).to_csv("Diseased Gut/%s_%s_landmark_matrix_PA.csv" %(pheno_a, pheno_b))

        S_et = 1 - (S_et / S_et.max())
        D_et = np.sqrt(S_et).astype(np.float32)
        pd.DataFrame(D_et, index = y[idx], columns = y[idx]).to_csv("Diseased Gut/%s_%s_et_matrix_PA.csv" %(pheno_a, pheno_b))

        S_rf = 1 - (S_rf / S_rf.max())
        D_rf = np.sqrt(S_rf).astype(np.float32)
        pd.DataFrame(D_rf, index = y[idx], columns = y[idx]).to_csv("Diseased Gut/%s_%s_rf_matrix_PA.csv" %(pheno_a, pheno_b))

        #Transform and plot
        E_lm = um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D_lm)

        stat = pcoa(pairwise_distances(E_lm, metric = "euclidean").astype(np.float32))
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        stat = permanova(DistanceMatrix(D_lm), y[idx])
        R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
        stat["R2"] = R2
        print(stat, R2)
        stat.to_csv("Diseased Gut/%s_%s_permanova_landmark_PA.csv" %(pheno_a, pheno_b))

        stat = stat.values

        title = "LANDMark (Oracle) (UMAP with PCoA)"

        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[1, 0])

        axs[1, 0].set_title(title)
        axs[1, 0].set_xlabel("PC 1 (%s Percent)" %pc1)
        axs[1, 0].set_ylabel("PC 2 (%s Percent)" %pc2)
        axs[1, 0].legend_ = None

        stat = pcoa(D_lm)
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        title = "LANDMark (Oracle) Similarity (PCoA)"
        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[1, 1])

        axs[1, 1].set_title(title)
        axs[1, 1].set_xlabel("PCoA 1 (%s Percent)" %pc1)
        axs[1, 1].set_ylabel("PCoA 2 (%s Percent)" %pc2)
        axs[1, 1].legend_ = None

        #Transform and plot
        E_et = um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D_et)

        stat = pcoa(pairwise_distances(E_et, metric = "euclidean").astype(np.float32))
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        stat = permanova(DistanceMatrix(D_et), y[idx])
        R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
        stat["R2"] = R2
        print(stat, R2)
        stat.to_csv("Diseased Gut/%s_%s_permanova_et_PA.csv" %(pheno_a, pheno_b))

        stat = stat.values

        title = "Extra Trees (UMAP with PCoA)"

        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[2, 0])

        axs[2, 0].set_title(title)
        axs[2, 0].set_xlabel("PC 1 (%s Percent)" %pc1)
        axs[2, 0].set_ylabel("PC 2 (%s Percent)" %pc2)
        axs[2, 0].legend_ = None

        stat = pcoa(D_et)
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        title = "Extra Trees Similarity (PCoA)"
        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[2, 1])

        axs[2, 1].set_title(title)
        axs[2, 1].set_xlabel("PCoA 1 (%s Percent)" %pc1)
        axs[2, 1].set_ylabel("PCoA 2 (%s Percent)" %pc2)
        axs[2, 1].legend_ = None

        #Transform and plot
        E_rf = um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D_rf)

        stat = pcoa(pairwise_distances(E_rf, metric = "euclidean").astype(np.float32))
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        stat = permanova(DistanceMatrix(D_rf), y[idx])
        R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
        stat["R2"] = R2
        print(stat, R2)
        stat.to_csv("Diseased Gut/%s_%s_permanova_rf_PA.csv" %(pheno_a, pheno_b))

        stat = stat.values

        title = "Random Forest (UMAP with PCoA)"

        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[3, 0])

        axs[3, 0].set_title(title)
        axs[3, 0].set_xlabel("PC 1 (%s Percent)" %pc1)
        axs[3, 0].set_ylabel("PC 2 (%s Percent)" %pc2)
        axs[3, 0].legend_ = None
     
        plt.tight_layout()
        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        plt.tight_layout()

        stat = pcoa(D_rf)
        pc1 = str(stat.proportion_explained[0] * 100)[0:5]
        pc2 = str(stat.proportion_explained[1] * 100)[0:5]
        E_ja = stat.samples.values

        title = "Random Forest Similarity (PCoA)"
        sns.scatterplot(x = E_ja[:, 0],
                        y = E_ja[:, 1],
                        hue = y[idx],
                        ax = axs[3, 1])

        axs[3, 1].set_title(title)
        axs[3, 1].set_xlabel("PCoA 1 (%s Percent)" %pc1)
        axs[3, 1].set_ylabel("PCoA 2 (%s Percent)" %pc2)

        plt.savefig("Diseased Gut/%s_%s_PA.svg" %(pheno_a, pheno_b))
        plt.close()

#Calculate correlations
unsupervised_comparison_2 = False
if unsupervised_comparison_2 == True:

    X = X_clr_t1
    y = y_t1
    
    n = np.sqrt(X_clr.shape[1]) / X_clr.shape[1]

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC", 
             "UC-HC"]

    from sklearn.metrics import pairwise_distances, balanced_accuracy_score
    from skbio.stats.distance import permanova, DistanceMatrix
    from skbio.stats.ordination import pcoa
    from sklearn.preprocessing import OneHotEncoder
    from scipy.stats import spearmanr

    for phenotypes in pheno:

        fig, axs = plt.subplots(4, 3)
        fig.set_figheight(11)
        fig.set_figwidth(8.5)

        #Transform and plot using just distance
        #Transforms
        D = pd.read_csv("Diseased Gut/PerMANOVA CLR/%s_distance_matrix_CLR.csv" %phenotypes.replace("-", "_"))
        y = np.asarray([entry.split(".")[0] for entry in list(D.columns.values[1:])])
        D = D.values[:, 1:]

        hue = []
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                tmp = [y[i], y[j]]
                tmp.sort()
                hue.append("-".join(tmp))

        D_orig = np.copy(D, "C").flatten()
        D_pcoa_original = pcoa(pairwise_distances(D, metric = "euclidean").astype(np.float32)).samples.values.flatten()
        D2 = pairwise_distances(um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D))
        D_umap = D2.astype(np.float32).flatten()
        D_umap_pcoa = pcoa(pairwise_distances(D2, metric = "euclidean").astype(np.float32)).samples.values
        D_umap_pcoa = pairwise_distances(D_umap_pcoa, metric = "euclidean").flatten()

        #Original vs PCoA
        rho, p_val = spearmanr(D_orig, D_pcoa_original)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_pcoa_original,
                        ax = axs[0, 0],
                        hue = hue,
                        s = 10)

        title = "A\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[0, 0].set_title(title)
        axs[0, 0].set_xlabel("Original Dissimilarities")
        axs[0, 0].set_ylabel("Projected Dissimilarities")

        #Original vs UMAP
        rho, p_val = spearmanr(D_orig, D_umap)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap,
                        ax = axs[0, 1],
                        hue = hue,
                        s = 10)

        title = "B\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[0, 1].set_title(title)
        axs[0, 1].set_xlabel("Original Dissimilarities")
        axs[0, 1].set_ylabel("Projected Dissimilarities")
        axs[0, 0].legend_ = None

        #Original vs UMAP w/ PCoA
        rho, p_val = spearmanr(D_orig, D_umap_pcoa)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap_pcoa,
                        ax = axs[0, 2],
                        hue = hue,
                        s = 10)

        title = "C\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[0, 2].set_title(title)
        axs[0, 2].set_xlabel("Original Dissimilarities")
        axs[0, 2].set_ylabel("Projected Dissimilarities")
        axs[0, 2].legend_ = None

        #Classifiy
        S_lm = pd.read_csv("Diseased Gut/PerMANOVA CLR/%s_landmark_matrix_CLR.csv" %phenotypes.replace("-", "_")).values[:, 1:]
        S_et = pd.read_csv("Diseased Gut/PerMANOVA CLR/%s_et_matrix_CLR.csv" %phenotypes.replace("-", "_")).values[:, 1:]
        S_rf = pd.read_csv("Diseased Gut/PerMANOVA CLR/%s_rf_matrix_clr.csv" %phenotypes.replace("-", "_")).values[:, 1:]

        #Transform and plot
        #Transforms
        D = np.copy(S_lm, "C")
        D_orig = np.copy(D, "C").flatten()
        D_pcoa_original = pcoa(pairwise_distances(D, metric = "euclidean").astype(np.float32)).samples.values.flatten()
        D2 = pairwise_distances(um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D))
        D_umap = D2.astype(np.float32).flatten()
        D_umap_pcoa = pcoa(pairwise_distances(D2, metric = "euclidean").astype(np.float32)).samples.values
        D_umap_pcoa = pairwise_distances(D_umap_pcoa, metric = "euclidean").flatten()

        #Original vs PCoA
        rho, p_val = spearmanr(D_orig, D_pcoa_original)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_pcoa_original,
                        ax = axs[1, 0],
                        hue = hue,
                        s = 10)

        title = "D\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[1, 0].set_title(title)
        axs[1, 0].set_xlabel("Original Dissimilarities")
        axs[1, 0].set_ylabel("Projected Dissimilarities")
        axs[1, 0].legend_ = None

        #Original vs UMAP
        rho, p_val = spearmanr(D_orig, D_umap)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap,
                        ax = axs[1, 1],
                        hue = hue,
                        s = 10)

        title = "E\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[1, 1].set_title(title)
        axs[1, 1].set_xlabel("Original Dissimilarities")
        axs[1, 1].set_ylabel("Projected Dissimilarities")
        axs[1, 1].legend_ = None

        #Original vs UMAP w/ PCoA
        rho, p_val = spearmanr(D_orig, D_umap_pcoa)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap_pcoa,
                        ax = axs[1, 2],
                        hue = hue,
                        s = 10)

        title = "F\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[1, 2].set_title(title)
        axs[1, 2].set_xlabel("Original Dissimilarities")
        axs[1, 2].set_ylabel("Projected Dissimilarities")
        axs[1, 2].legend_ = None

        #Transform and plot
        #Transforms
        D = np.copy(S_et, "C")
        D_orig = np.copy(D, "C").flatten()
        D_pcoa_original = pcoa(pairwise_distances(D, metric = "euclidean").astype(np.float32)).samples.values.flatten()
        D2 = pairwise_distances(um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D))
        D_umap = D2.astype(np.float32).flatten()
        D_umap_pcoa = pcoa(pairwise_distances(D2, metric = "euclidean").astype(np.float32)).samples.values
        D_umap_pcoa = pairwise_distances(D_umap_pcoa, metric = "euclidean").flatten()

        #Original vs PCoA
        rho, p_val = spearmanr(D_orig, D_pcoa_original)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_pcoa_original,
                        ax = axs[2, 0],
                        hue = hue,
                        s = 10)

        title = "G\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[2, 0].set_title(title)
        axs[2, 0].set_xlabel("Original Dissimilarities")
        axs[2, 0].set_ylabel("Projected Dissimilarities")
        axs[2, 0].legend_ = None

        #Original vs UMAP
        rho, p_val = spearmanr(D_orig, D_umap)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap,
                        ax = axs[2, 1],
                        hue = hue,
                        s = 10)

        title = "H\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[2, 1].set_title(title)
        axs[2, 1].set_xlabel("Original Dissimilarities")
        axs[2, 1].set_ylabel("Projected Dissimilarities")
        axs[2, 1].legend_ = None

        #Original vs UMAP w/ PCoA
        rho, p_val = spearmanr(D_orig, D_umap_pcoa)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap_pcoa,
                        ax = axs[2, 2],
                        hue = hue,
                        s = 10)

        title = "I\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[2, 2].set_title(title)
        axs[2, 2].set_xlabel("Original Dissimilarities")
        axs[2, 2].set_ylabel("Projected Dissimilarities")
        axs[2, 2].legend_ = None

        #Transform and plot
        #Transforms
        D = np.copy(S_rf, "C")
        D_orig = np.copy(D, "C").flatten()
        D_pcoa_original = pcoa(pairwise_distances(D, metric = "euclidean").astype(np.float32)).samples.values.flatten()
        D2 = pairwise_distances(um.UMAP(n_components = 30, metric = "precomputed", min_dist = 0.01).fit_transform(D))
        D_umap = D2.astype(np.float32).flatten()
        D_umap_pcoa = pcoa(pairwise_distances(D2, metric = "euclidean").astype(np.float32)).samples.values
        D_umap_pcoa = pairwise_distances(D_umap_pcoa, metric = "euclidean").flatten()

        #Original vs PCoA
        rho, p_val = spearmanr(D_orig, D_pcoa_original)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_pcoa_original,
                        ax = axs[3, 0],
                        hue = hue,
                        s = 10)

        title = "J\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[3, 0].set_title(title)
        axs[3, 0].set_xlabel("Original Dissimilarities")
        axs[3, 0].set_ylabel("Projected Dissimilarities")
        axs[3, 0].legend_ = None

        #Original vs UMAP
        rho, p_val = spearmanr(D_orig, D_umap)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap,
                        ax = axs[3, 1],
                        hue = hue,
                        s = 10)

        title = "K\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[3, 1].set_title(title)
        axs[3, 1].set_xlabel("Original Dissimilarities")
        axs[3, 1].set_ylabel("Projected Dissimilarities")
        axs[3, 1].legend_ = None

        #Original vs UMAP w/ PCoA
        rho, p_val = spearmanr(D_orig, D_umap_pcoa)
        rho = str(rho)[0:6]
        if p_val <= 0.001:
            p_val = str("Less than 0.001")
        else:
            p_val = str(p_val)[0:6]

        sns.scatterplot(x = D_orig,
                        y = D_umap_pcoa,
                        ax = axs[3, 2],
                        hue = hue,
                        s = 10)

        title = "L\nSpearman's rho: %s\np_value: %s" %(rho, p_val)

        axs[3, 2].set_title(title)
        axs[3, 2].set_xlabel("Original Dissimilarities")
        axs[3, 2].set_ylabel("Projected Dissimilarities")
        axs[3, 2].legend_ = None

        plt.tight_layout()
        plt.savefig("Diseased Gut/%s_PA_corr.svg" %phenotypes)
        plt.close()

    fdfd = 5

#Calculate data that shows effect size vs features considered
unsupervised_comparison_3 = False
if unsupervised_comparison_3 == True:

    X = X_clr_t1
    y = y_t1
    
    n = np.sqrt(X_clr.shape[1]) / X_clr.shape[1]
    max_mult = [1, 2, 4, 8, 16]

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC",
             "UC-HC"]

    R_dict = {}
    F_dict = {}
    P_dict = {}

    from sklearn.metrics import pairwise_distances, balanced_accuracy_score
    from skbio.stats.distance import permanova, DistanceMatrix
    from skbio.stats.ordination import pcoa
    from sklearn.preprocessing import OneHotEncoder

    for phenotypes in pheno:

        R_dict[phenotypes] = {1: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              2: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              4: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              8: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              16: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []}}

        F_dict[phenotypes] = {1: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              2: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              4: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              8: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              16: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []}}

        P_dict[phenotypes] = {1: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              2: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              4: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              8: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []},
                              16: {"Distance": [], "LANDMark": [], "ETC": [], "RF": []}}

        for n_mult in max_mult:

            pheno_a, pheno_b = phenotypes.split("-")

            idx = np.where(((y == pheno_a) | (y == pheno_b)), True, False)

            #Transform just distance
            X_1 = X[idx]

            D_ja = pairwise_distances(X_1, metric = "euclidean") #jaccard/euclidean
            for rep in range(10):
                stat = permanova(DistanceMatrix(D_ja.astype(np.float32)), y[idx])
                R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
                R_dict[phenotypes][n_mult]["Distance"].append(R2)
                F_dict[phenotypes][n_mult]["Distance"].append(stat.values[4])
                P_dict[phenotypes][n_mult]["Distance"].append(stat.values[5])

            #Classifiy
            for rep in range(10):
                S_lm = np.zeros((X[idx].shape[0], X[idx].shape[0]))
                S_et = np.zeros((X[idx].shape[0], X[idx].shape[0]))
                S_rf = np.zeros((X[idx].shape[0], X[idx].shape[0]))
                for i in range(30):
                    #Create a random labeled set of data
                    print("Iteration:", i+1, "N", n_mult, "Repeat", rep)

                    clr_T = True
                    if clr_T == True:
                        X_new, y_new = addcl1(X_orig_t1[idx], y[idx])
                        X_new = clr(multiplicative_replacement(closure(X_new)))

                    else:
                        X_new, y_new = addcl1(X[idx], y[idx])

                    X_tr, X_te, y_tr, y_te = train_test_split(X_new, y_new, test_size = 0.2, stratify = y_new)

                    model_lm = LANDMarkClassifier(n_jobs = 5, max_features = n*n_mult).fit(X_tr.astype(np.float), y_tr)
                    model_et = ExtraTreesClassifier(max_features = n*n_mult).fit(X_tr.astype(np.float), y_tr)
                    model_rf = RandomForestClassifier(max_features = n*n_mult).fit(X_tr.astype(np.float), y_tr)

                    print(balanced_accuracy_score(y_te, model_lm.predict(X_te)),
                            balanced_accuracy_score(y_te, model_et.predict(X_te)),
                            balanced_accuracy_score(y_te, model_rf.predict(X_te)))

                    #LANDMark Proximity
                    leaves = model_lm.proximity(X[idx].astype(np.float))
                    M = np.dot(leaves, leaves.T)
                    M =  M / np.max(M)
                    S_lm += M

                    #Random Forest/Extra Trees Proximity
                    leaves = model_et.apply(X[idx])
                    M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
                    M = np.dot(M, M.T)
                    M = M / M.max()
                    S_et += M

                    leaves = model_rf.apply(X[idx])
                    M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
                    M = np.dot(M, M.T)
                    M = M / M.max()
                    S_rf += M

                S_lm = 1 - (S_lm / S_lm.max())
                D_lm = np.sqrt(S_lm).astype(np.float32)

                S_et = 1 - (S_et / S_et.max())
                D_et = np.sqrt(S_et).astype(np.float32)

                S_rf = 1 - (S_rf / S_rf.max())
                D_rf = np.sqrt(S_rf).astype(np.float32)

                #Calculate statistics
                stat = permanova(DistanceMatrix(D_lm), y[idx])
                R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
                R_dict[phenotypes][n_mult]["LANDMark"].append(R2)
                F_dict[phenotypes][n_mult]["LANDMark"].append(stat.values[4])
                P_dict[phenotypes][n_mult]["LANDMark"].append(stat.values[5])

                stat = permanova(DistanceMatrix(D_et), y[idx])
                R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
                R_dict[phenotypes][n_mult]["ETC"].append(R2)
                F_dict[phenotypes][n_mult]["ETC"].append(stat.values[4])
                P_dict[phenotypes][n_mult]["ETC"].append(stat.values[5])

                stat = permanova(DistanceMatrix(D_rf), y[idx])
                R2 = 1 - 1 / (1 + stat[4] * stat[4] / (stat[2] - stat[3] - 1))
                R_dict[phenotypes][n_mult]["RF"].append(R2)
                F_dict[phenotypes][n_mult]["RF"].append(stat.values[4])
                P_dict[phenotypes][n_mult]["RF"].append(stat.values[5])

            csv_name = "%s_%s_R.csv" %(phenotypes, str(n_mult))
            pd.DataFrame.from_dict(R_dict[phenotypes][n_mult]).to_csv(csv_name)
            csv_name = "%s_%s_F.csv" %(phenotypes, str(n_mult))
            pd.DataFrame.from_dict(F_dict[phenotypes][n_mult]).to_csv(csv_name)
            csv_name = "%s_%s_P.csv" %(phenotypes, str(n_mult))
            pd.DataFrame.from_dict(P_dict[phenotypes][n_mult]).to_csv(csv_name)

#Select ASVs
feature_select = False
if feature_select == True:
    import shap as sh
    from sklearn.metrics import make_scorer, balanced_accuracy_score

    class LANDMarkWrapper(BaseEstimator, ClassifierMixin):

        def __init__(self):
            
            return None

        def fit(self, X, y):

            n_step = np.sqrt(X.shape[1]) / X.shape[1]
            #n_step = float(X.shape[1]) * 0.8 #Default

            self.model = LANDMarkClassifier(128, n_jobs = 6, max_features = n_step)

            self.model.fit(X, y)

            self.coef_ = self.model.feature_importances_

            return self

        def predict(self, X):

            return self.model.predict(X)

        def predict_proba(self, X):

            return self.model.predict_proba(X)

    n = np.sqrt(X_clr.shape[1]) / X_clr.shape[1]

    taxa_dict = {entry[0]:entry[1:] for entry in taxa_tab}

    score_fun = make_scorer(balanced_accuracy_score)

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC", 
             "UC-HC"]

    for phenotypes in pheno:

        ac_scores = np.zeros(shape = (30, 3))
        selected_asvs = []

        pheno_a, pheno_b = phenotypes.split("-")

        idx = np.where(((y_t1 == pheno_a) | (y_t1 == pheno_b)), True, False)

        from sklearn.utils import resample

        for i in range(30):

            X_train, X_test, y_train, y_test = train_test_split(X_clr_t1[idx],
                                                                y_t1[idx], 
                                                                test_size = 0.5, stratify = y_t1[idx], #test size originally 50
                                                                random_state = i)

            #Base classifier
            clf_base = LANDMarkClassifier(128, n_jobs = 6, max_features = n)
            #clf_base = ExtraTreesClassifier(128)
            #clf_base = RandomForestClassifier(128)# 
            clf_base.fit(X_train, y_train)

            score_a = balanced_accuracy_score(y_test, clf_base.predict(X_test))
            ac_scores[i, 0] = score_a
            print(phenotypes, i, score_a)

            select_features = True
            if select_features == True:
                #RFE
                clf = RFE(LANDMarkWrapper(), n_features_to_select = 200, step = 0.2, verbose =0).fit(X_train, y_train)
                #clf = RFE(ExtraTreesClassifier(128), n_features_to_select = 200, step = 0.2, verbose =0).fit(X_train, y_train)
                #clf = RFE(RandomForestClassifier(128), n_features_to_select = 200, step = 0.2, verbose =0).fit(X_train, y_train)
                print(balanced_accuracy_score(y_test, clf.predict(X_test)))
                new_clf = RFECV(LANDMarkWrapper(),
                                #ExtraTreesClassifier(128),
                                #RandomForestClassifier(128),
                                                   step = 0.05,
                                                   min_features_to_select = 20,
                                                   cv = 5,
                                                   scoring = score_fun,
                                                   verbose = 0,
                                 #                  n_jobs = 5
                                                   ).fit(clf.transform(X_train), y_train)

                probs = new_clf.predict_proba(clf.transform(X_test))

                predictions = new_clf.predict(clf.transform(X_test))
                incorrect_predictions = np.asarray([False if predictions[i] == y_test[i] else True for i in range(predictions.shape[0])])

                score_b = balanced_accuracy_score(y_test, new_clf.predict(clf.transform(X_test)))
                ac_scores[i, 1] = score_b

                print(i, ":", score_a, score_b)

                new_ASVs = new_clf.transform(clf.transform([ASVs[X_feat]]))[0]
                new_ASVs = ["%s (%s)" %(taxa_dict[ASV][16], ASV) for ASV in new_ASVs]

                ac_scores[i, 2] = i

                new_ASVs.append(str(i))
                selected_asvs.append(new_ASVs)

        ac_scores = pd.DataFrame(ac_scores, index = None, columns = ["Before", "After", "Fold"])
        ac_scores.to_csv("Diseased Gut/%s/%s_scores_lm_500_80.csv" %(phenotypes, phenotypes))

        #selected_asvs = pd.DataFrame(selected_asvs)
        #selected_asvs.to_csv("Diseased Gut/%s/%s_ASVs_lm.csv" %(phenotypes, phenotypes))

    fdfd = 5

#Bayesian t-tests
compare_runs = False
if compare_runs == True:

    from baycomp import two_on_single

    comparison = []

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC", 
             "UC-HC"]

    clfs = ["lm", "et", "rf"]

    for phenotypes in pheno:
        for i in range(0, 3-1):
            a = pd.read_csv("Diseased Gut/%s/%s_scores_%s_clr.csv" %(phenotypes, phenotypes, clfs[i]))

            #Calculate Before/After statistics for model i
            model = two_on_single(a["Before"].values, a["After"].values, rope = 0.025)
            comparison.append([phenotypes, "%s-%s-Before-After" %(clfs[i], clfs[i]), 
                               np.mean(a["Before"].values), np.std(a["Before"].values, ddof = 1), 
                               np.mean(a["After"].values), np.std(a["After"].values, ddof = 1), 
                               model[0], model[1], model[2]])
            plt.show()
            plt.close()

            for j in range(i+1, 3):
                b = pd.read_csv("Diseased Gut/%s/%s_scores_%s_clr.csv" %(phenotypes, phenotypes, clfs[j]))

                #Calculate Before/After statistics for model j
                model = two_on_single(b["Before"].values, b["After"].values, rope = 0.025)
                comparison.append([phenotypes, "%s-%s-Before-After" %(clfs[j], clfs[j]), 
                                   np.mean(b["Before"].values), np.std(b["Before"].values, ddof = 1), 
                                   np.mean(b["After"].values), np.std(b["After"].values, ddof = 1), 
                                   model[0], model[1], model[2]])


                #Calculate Generalization Performance Before RFE
                model = two_on_single(a["Before"].values, b["Before"].values, rope = 0.025)
                comparison.append([phenotypes, "%s-%s-Before" %(clfs[i], clfs[j]), 
                                   np.mean(a["Before"].values), np.std(a["Before"].values, ddof = 1), 
                                   np.mean(b["Before"].values), np.std(b["Before"].values, ddof = 1), 
                                   model[0], model[1], model[2]])


                #Calculate Generalization Performance After RFE
                model = two_on_single(a["After"].values, b["After"].values, rope = 0.025)
                comparison.append([phenotypes, "%s-%s-After" %(clfs[i], clfs[j]), 
                                   np.mean(a["After"].values), np.std(a["After"].values, ddof = 1), 
                                   np.mean(b["After"].values), np.std(b["After"].values, ddof = 1), 
                                   model[0], model[1], model[2]])

    comparison = pd.DataFrame(comparison, index = None, columns = ["Sampling Location", "Comparison (A-B)", 
                                                                   "Mean A", "Std Dev A", "Mean B", "Std Dev B", "A > B", "A=B", "B>A"])
    print(comparison)
    comparison.to_csv("Diseased Gut/stat_1_clr.csv")

    fdfd = 5

#Bayesian t-tests
compare_runs_pa_clr = False
if compare_runs_pa_clr == True:

    from baycomp import two_on_single

    comparison = []

    pheno = ["CD-HC",
             "RA-HC",
             "MS-HC", 
             "UC-HC"]

    clfs = ["lm", "et", "rf"]

    for phenotypes in pheno:
        for i in range(0, 3):
            a = pd.read_csv("Diseased Gut/%s/%s_scores_%s.csv" %(phenotypes, phenotypes, clfs[i]))
            b = pd.read_csv("Diseased Gut/%s/%s_scores_%s_clr.csv" %(phenotypes, phenotypes, clfs[i]))

            #Calculate PA/CLR statistics for model i
            model = two_on_single(a["Before"].values, b["Before"].values, rope = 0.025)
            comparison.append([phenotypes, "%s-%s-PA-CLR" %(clfs[i], clfs[i]), np.mean(a["Before"].values), np.std(a["Before"].values, ddof = 1), np.mean(b["Before"].values), np.std(b["Before"].values, ddof = 1), model[0], model[1], model[2]])

    comparison = pd.DataFrame(comparison, index = None, columns = ["Sampling Location", "Comparison (A-B)", 
                                                                   "Mean A", "Std Dev A", "Mean B", "Std Dev B", "A > B", "A=B", "B>A"])
    print(comparison)
    comparison.to_csv("Diseased Gut/stat_2.csv")

    fdfd = 5

#Shapley plots
create_graphs = True
if create_graphs == True:

    import shap as sh
    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from math import ceil

    selASVs = "Blautia (Zotu1)	Collinsella (Zotu2)	Bifidobacterium (Zotu5)	Dorea (Zotu15)	Faecalibacillus (Zotu17)	Gemmiger (Zotu18)	Bifidobacterium (Zotu23)	Blautia (Zotu27)	Anaerobutyricum (Zotu30)	Dorea (Zotu34)	Blautia (Zotu36)	Faecalimonas (Zotu37)	Lachnospiracea_incertae_sedis (Zotu39)	Streptococcus (Zotu41)	Senegalimassilia (Zotu43)	Enterococcus (Zotu44)	Terrisporobacter (Zotu45)	Veillonella (Zotu47)	Bifidobacterium (Zotu48)	Faecalibacterium (Zotu53)	Neglecta (Zotu54)	Turicibacter (Zotu58)	Ligilactobacillus (Zotu60)	Eggerthella (Zotu62)	Coprococcus (Zotu64)	Lactococcus (Zotu65)	Mediterraneibacter (Zotu69)	Slackia (Zotu70)	Dialister (Zotu77)	Bifidobacterium (Zotu80)	Holdemanella (Zotu83)	Faecalibacterium (Zotu86)	Blautia (Zotu92)	Blautia (Zotu95)	Dorea (Zotu97)	Monoglobus (Zotu102)	Streptococcus (Zotu107)	Gordonibacter (Zotu113)	Terrisporobacter (Zotu114)	Faecalibacterium (Zotu118)	Blautia (Zotu123)	Coprococcus (Zotu125)	Longicatena (Zotu126)	Neglecta (Zotu135)	Lacticaseibacillus (Zotu145)	Senegalimassilia (Zotu146)	Bifidobacterium (Zotu147)	Collinsella (Zotu156)	Mediterraneibacter (Zotu171)	Blautia (Zotu173)	Clostridium_sensu_stricto (Zotu175)	Ruminococcus (Zotu180)	Roseburia (Zotu182)	Blautia (Zotu187)	Lachnospiracea_incertae_sedis (Zotu189)	Clostridium_IV (Zotu197)	Clostridium_XlVb (Zotu198)	Intestinimonas (Zotu207)	Ihubacter (Zotu215)	Faecalibacterium (Zotu218)	Ihubacter (Zotu219)	Coprococcus (Zotu230)	Mediterraneibacter (Zotu243)	Gordonibacter (Zotu247)	Enterocloster (Zotu251)	Faecalibacterium (Zotu265)	Ihubacter (Zotu267)	Mediterraneibacter (Zotu274)	Anaeromassilibacillus (Zotu282)	Ruminococcus (Zotu283)	Clostridium_sensu_stricto (Zotu284)	Clostridium_sensu_stricto (Zotu289)	Gemella (Zotu302)	Adlercreutzia (Zotu311)	Raoultibacter (Zotu312)	Enteroscipio (Zotu315)	Phascolarctobacterium (Zotu316)	Longicatena (Zotu319)	Ruminococcus (Zotu335)	Eubacterium (Zotu340)	Neglecta (Zotu347)	Anaerococcus (Zotu353)	Finegoldia (Zotu377)	Neglecta (Zotu380)	Ihubacter (Zotu398)	Colidextribacter (Zotu411)	Massilimicrobiota (Zotu416)	Staphylococcus (Zotu420)	Colidextribacter (Zotu458)	Fenollaria (Zotu485)	Raoultibacter (Zotu499)	Actinomyces (Zotu535)	Rothia (Zotu605)	Dialister (Zotu633)	Ruminococcus (Zotu657)	Lawsonibacter (Zotu667)	Massilimicrobiota (Zotu713)	Parvimonas (Zotu809)	Streptococcus (Zotu996)	Paraclostridium (Zotu1537)"
    selASVs = selASVs.strip("\n").split("\t")
    ASV_names = [x.replace("Zotu", "ASV") for x in selASVs if "Zotu" in x]
    ASV_names = [x.split(" ")[1].strip("(").strip(")") for x in ASV_names]
    features = [x.replace("ASV", "Zotu") for x in ASV_names]
    
    taxa = selPHYLA = {entry[0]: entry[17] for entry in taxa_tab}
    taxa = np.asarray(["%s (%s)" %(taxa[entry], ASV_names[i].replace("ASV", "ASV ")) for i, entry in enumerate(features)])

    X_ss = pd.DataFrame(X_clr_t2, columns = list(ASVs[X_feat]))[features]

    selPHYLA = {entry[0]: entry[5] for entry in taxa_tab}
    P = list({selPHYLA[key] for key in features})

    idx = np.where(((y_t2 == "CD") | (y_t2 == "HC")), True, False)

    X_ss = X_ss[idx].values
    y = y_t1[idx]

    X_train, X_test, y_train, y_test = train_test_split(X_ss,
                                                        y, test_size = 0.5, stratify = y, random_state = 7)
    
    n = np.sqrt(X_train.shape[1]) / X_train.shape[1]
    from scipy.stats import pearsonr
    clf = LANDMarkClassifier(128, n_jobs = 8, max_features=n)
    #clf = ExtraTreesClassifier(128)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    score_b = balanced_accuracy_score(y_test, clf.predict(X_test))

    explainer = sh.Explainer(clf.predict_proba,
                        sh.maskers.Independent(X_train),
                        feature_names = ASV_names,
                        output_names = clf.classes_,
                        algorithm = "permutation")
            
    scores = explainer(X_test, 
                       max_evals = 4097)
    
    score_avg = np.abs(scores.values[:, :, 0]).mean(axis = 0)
    score_mean = np.mean(score_avg)
    score_avg = score_avg > score_mean

    y_order = np.asarray([p if predictions[j] == y_test[j] else "%s*" %p for j, p in enumerate(predictions)])
    lut = dict(zip(["CD", "HC"], "bg"))

    lut2 = dict(zip(P, "gy"))
    P2 = [selPHYLA[key] for key in features]

    a = sns.clustermap(pd.DataFrame(scores.values[:, :, 0][:, score_avg], index = y_order, columns = taxa[score_avg]),
                   method = "ward",
                   metric = "euclidean",
                   row_cluster = True,
                   col_cluster = True,
                   xticklabels = True,
                   yticklabels = True,
                   cmap = sh.plots.colors.red_blue,
                   row_colors = np.asarray([lut[entry.strip("*")] for entry in y_order]),
                   col_colors = np.asarray([lut2[entry] for entry in P2])[score_avg])
    
    col_link = a.dendrogram_col.reordered_ind
    row_link = a.dendrogram_row.reordered_ind

    sns.clustermap(pd.DataFrame(X_test[:, score_avg][row_link][:, col_link], index = y_order[row_link], columns = taxa[score_avg][col_link]), 
                cmap = sh.plots.colors.red_blue,
                row_cluster = False,
                col_cluster = False,
                xticklabels = True, yticklabels = True,
                row_colors = [lut[entry.strip("*")] for entry in y_order[row_link]])

    plt.close()

#Decision boundary plot
create_pcoa = False
if create_pcoa == True:

    import shap as sh
    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from skbio.stats.ordination import pcoa
    from math import ceil

    selASVs = "Blautia (Zotu1),Collinsella (Zotu2),Bifidobacterium (Zotu5),Anaerobutyricum (Zotu7),Bifidobacterium (Zotu9),Gemmiger (Zotu18),Mediterraneibacter (Zotu22),Bifidobacterium (Zotu23),Anaerostipes (Zotu25),Blautia (Zotu26),Blautia (Zotu27),Adlercreutzia (Zotu28),Streptococcus (Zotu35),Blautia (Zotu36),Faecalimonas (Zotu37),Clostridium sensu stricto (Zotu38),Lachnospiracea_incertae_sedis (Zotu39),Streptococcus (Zotu41),Senegalimassilia (Zotu43),Enterococcus (Zotu44),Terrisporobacter (Zotu45),Bifidobacterium (Zotu48),Veillonella (Zotu50),Neglecta (Zotu54),Catenibacterium (Zotu57),Turicibacter (Zotu58),Ligilactobacillus (Zotu60),Eggerthella (Zotu62),Coprococcus (Zotu64),Faecalibacillus (Zotu67),Erysipelatoclostridium (Zotu68),Mediterraneibacter (Zotu69),Slackia (Zotu70),Clostridium sensu stricto (Zotu72),Dialister (Zotu77),Bifidobacterium (Zotu80),Holdemanella (Zotu83),Faecalibacterium (Zotu86),Blautia (Zotu92),Blautia (Zotu95),Dorea (Zotu97),Blautia (Zotu100),Slackia (Zotu101),Monoglobus (Zotu102),Streptococcus (Zotu107),Turicibacter (Zotu111),Mogibacterium (Zotu112),Gordonibacter (Zotu113),Coprococcus (Zotu125),Longicatena (Zotu126),Lachnospiracea_incertae_sedis (Zotu133),Clostridium IV (Zotu134),Coprococcus (Zotu143),Collinsella (Zotu144),Lacticaseibacillus (Zotu145),Roseburia (Zotu152),Clostridium XVIII (Zotu164),Blautia (Zotu173),Roseburia (Zotu182),Blautia (Zotu186),Blautia (Zotu187),Lachnospiracea_incertae_sedis (Zotu189),Lactobacillus (Zotu191),Clostridium IV (Zotu197),Clostridium XlVb (Zotu198),Blautia (Zotu200),Peptococcus (Zotu203),Intestinimonas (Zotu207),Erysipelatoclostridium (Zotu213),Ihubacter (Zotu215),Faecalibacterium (Zotu218),Ihubacter (Zotu219),Streptococcus (Zotu221),Coprococcus (Zotu230),Lachnospiracea_incertae_sedis (Zotu232),Mediterraneibacter (Zotu243),Gordonibacter (Zotu247),Anaeromassilibacillus (Zotu282),Gemella (Zotu302),Adlercreutzia (Zotu311),Enteroscipio (Zotu315),Phascolarctobacterium (Zotu316),Ruminococcus (Zotu335),Anaerostipes (Zotu346),Schaalia (Zotu352),Anaerococcus (Zotu353),Anaerofustis (Zotu362),Finegoldia (Zotu377),Streptococcus (Zotu408),Colidextribacter (Zotu411),Ruminococcus (Zotu429),Anaeromassilibacillus (Zotu443),Colidextribacter (Zotu458),Blautia (Zotu466),Raoultibacter (Zotu499),Raoultibacter (Zotu501),Actinomyces (Zotu505),Lachnospira (Zotu533),Corynebacterium (Zotu544),Holdemania (Zotu589),Peptoniphilus (Zotu626),Dialister (Zotu633),Ruminococcus (Zotu657),Peptoniphilus (Zotu694),Fournierella (Zotu717),Christensenella (Zotu787),Parvimonas (Zotu809),Streptococcus (Zotu996),Streptococcus (Zotu1142),Holdemania (Zotu1154)"
    selASVs = selASVs.strip("\n").split(",")
    ASV_names = [x.replace("Zotu", "ASV ") for x in selASVs if "Zotu" in x]
    features = [entry.split()[-1].strip("(").strip(")") for entry in selASVs]

    X_pa = pd.DataFrame(X_clr_t1, columns = list(ASVs[X_feat]))[features].values

    idx = np.where(((y_t1 == "CD") | (y_t1 == "HC")), True, False)

    X_pa = X_clr_t1[idx]
    y = y_t1[idx]

    X_train, X_test, y_train, y_test = train_test_split(X_pa,
                                                        y, test_size = 0.5, stratify = y, random_state = 7)

    n = np.sqrt(X_train.shape[1]) / X_train.shape[1]

    clf_1 = LANDMarkClassifier(128, n_jobs = 8, max_features = n).fit(X_train, y_train)

    clf_2 = ExtraTreesClassifier(128).fit(X_train, y_train)

    clf_3 = RandomForestClassifier(128).fit(X_train, y_train)

    #LANDMark Proximity
    leaves = clf_1.proximity(X_test)
    M = np.dot(leaves, leaves.T)
    S_lm =  M / np.max(M)
    D_lm = 1 - S_lm
    D_lm = np.sqrt(D_lm)
    D_pcoa_lm = pcoa(D_lm)
    
    n_comp = D_pcoa_lm.eigvals.shape[0]
    for i in range(0, 10 - 1):
        for j in range(i + 1, 10):
            prop_expl_1 = str(D_pcoa_lm.proportion_explained["PC%s" %str(i+1)] * 100)[0:5] + "%"
            prop_expl_2 = str(D_pcoa_lm.proportion_explained["PC%s" %str(j+1)] * 100)[0:5] + "%"

            ax_1 = "PCoA %s (%s)" %(str(i + 1), prop_expl_1)
            ax_2 = "PCoA %s (%s)" %(str(j + 1), prop_expl_2)

            x_pc = D_pcoa_lm.samples["PC%s" %str(i+1)].values
            y_pc = D_pcoa_lm.samples["PC%s" %str(j+1)].values

            sns.scatterplot(x = x_pc,
                            y = y_pc,
                            hue = y_test)

            plt.xlabel(ax_1)
            plt.ylabel(ax_2)
            plt.legend("")

            plt.tight_layout()
            plt.savefig("Diseased Gut/CD-HC/LM/%s-%s_lm.svg" %("PC%s" %str(i+1), "PC%s" %str(j+1)))
            plt.close()

    #Random Forest/Extra Trees Proximity
    leaves = clf_2.apply(X_test)
    M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
    M = np.dot(M, M.T)
    S_et = M / M.max()
    D_et = 1 - S_et
    D_et = np.sqrt(D_et)
    D_pcoa_et = pcoa(D_et)

    n_comp = D_pcoa_et.eigvals.shape[0]
    for i in range(0, 10 - 1):
        for j in range(i + 1, 10):
            prop_expl_1 = str(D_pcoa_et.proportion_explained["PC%s" %str(i+1)] * 100)[0:5] + "%"
            prop_expl_2 = str(D_pcoa_et.proportion_explained["PC%s" %str(j+1)] * 100)[0:5] + "%"

            ax_1 = "PCoA %s (%s)" %(str(i + 1), prop_expl_1)
            ax_2 = "PCoA %s (%s)" %(str(j + 1), prop_expl_2)

            x_pc = D_pcoa_et.samples["PC%s" %str(i+1)].values
            y_pc = D_pcoa_et.samples["PC%s" %str(j+1)].values

            sns.scatterplot(x = x_pc,
                            y = y_pc,
                            hue = y_test)

            plt.xlabel(ax_1)
            plt.ylabel(ax_2)
            plt.legend("")
            plt.tight_layout()
            plt.savefig("Diseased Gut/CD-HC/ET/%s-%s_et.svg" %("PC%s" %str(i+1), "PC%s" %str(j+1)))
            plt.close()

    leaves = clf_3.apply(X_test)
    M = OneHotEncoder().fit(leaves).transform(leaves).toarray()
    M = np.dot(M, M.T)
    S_rf = M / M.max()
    D_rf = 1 - S_rf
    D_rf = np.sqrt(D_rf)
    D_pcoa_rf = pcoa(D_rf)

    n_comp = D_pcoa_rf.eigvals.shape[0]
    for i in range(0, 10 - 1):
        for j in range(i + 1, 10):
            prop_expl_1 = str(D_pcoa_rf.proportion_explained["PC%s" %str(i+1)] * 100)[0:5] + "%"
            prop_expl_2 = str(D_pcoa_rf.proportion_explained["PC%s" %str(j+1)] * 100)[0:5] + "%"

            ax_1 = "PCoA %s (%s)" %(str(i + 1), prop_expl_1)
            ax_2 = "PCoA %s (%s)" %(str(j + 1), prop_expl_2)

            x_pc = D_pcoa_rf.samples["PC%s" %str(i+1)].values
            y_pc = D_pcoa_rf.samples["PC%s" %str(j+1)].values

            sns.scatterplot(x = x_pc,
                            y = y_pc,
                            hue = y_test)

            plt.xlabel(ax_1)
            plt.ylabel(ax_2)

            if i > 0 and j > 3:
                pass
            else:
                plt.legend("")

            plt.tight_layout()
            plt.savefig("Diseased Gut/CD-HC/RF/%s-%s_rf.svg" %("PC%s" %str(i+1), "PC%s" %str(j+1)))
            plt.close()

    fdfd =51