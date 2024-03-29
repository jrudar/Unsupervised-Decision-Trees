# Unsupervised-Decision-Trees

Data and code to support our paper titled: "Decision Tree Ensembles Utilizing Multivariate Splits Are Effective at Investigating Beta-Diversity in Medically Relevant 16S Amplicon Sequencing Data"
https://www.biorxiv.org/content/10.1101/2022.03.31.486647v2

# Requirements
    To use the workflow LANDMark and TreeOrdination must be installed:
    
        LANDMark: https://github.com/jrudar/LANDMark
        TreeOrdination: https://github.com/jrudar/TreeOrdination

# Input
    To run the code, you will need the following files produced by the MetaWorks (https://github.com/terrimporter/MetaWorks) pipeline: 
        1) The ASV table (ESV.table)
        2) The Taxonomic Ranks and Confidence Scores of Each ASV (rdp.out.tmp)
        
    In addition you will need the file containing the metadata.
    
    These files are provided in this repository.
    
    Also, please ensure that all paths are correct on your computer.        

# Files
    control_get_data.py:
        Description: Creates synthetic data and analyzes the results (Martino et al. 2019).
        
        Section 'run_exp': Calculates balanced accuracy scores and PerMANOVA results for each experiment.
        Section 'get_stats_perm': Plots the PerMANOVA results
        Section 'get_stats_bacc': Plots the Balanced Accuracy Score Results
        
    diseased_gut_data.py:
        Description: Runs the experiment using the Crohn's Disease Data (Forbes et al. 2018).
        
        Section 'large_test': Calculates balanced accuracy scores and PerMANOVA results for each experiment
        Section 'feature_importance_test': Creates a TreeOrdination projection and calculates feature importance scores
        Section 'get_stats_perm': Plots the PerMANOVA results
        Section 'get_stats_bacc': Plots the Balanced Accuracy Score Results    
        
    baxter_data.py:
        Description: Calculates ROC-AUC scores and Balanced Accuracy Scores using the Baxter et al. (2016) Data
        
    baxter_data_fi.py:
        Description: Runs the experiment using the Baxter et al. (2016) data which calculates the sensitivity,
                     specificity, for the Normal vs. Cancer and Normal vs. Lesion comparisons. Calculates
                     Shapley scores.
                     
        Section 'get_stats': Plots ROC-AUC and Balanced Accuracy Scores
        Section 'normal_v_cancer': Calculates and saves CSV file of probability of cancer for each sample
        Section 'normal_v_cancer_stat': Calculates discrimination threshold, sensitivty, and specificity for
                                        the "normal_v_cancer" comparison
        Section 'normal_v_lesion': Calculates and saves CSV file of probability of lesion for each sample
        Section 'normal_v_lesion_stat': Calculates discrimination threshold, sensitivty, and specificity for
                                        the "normal_v_lesion" comparison
        Section 'get_fi_scores': Creates an TreeOrdination projection and calculates feature importance scores

# Create the Environment
    conda create -n TreeOrdFinal python=3.10

    conda activate TreeOrdFinal
        
    git clone https://github.com/jrudar/LANDMark
    cd LANDMark
    python setup.py sdist
    cd dist
    pip install LANDMark-1.2.0.tar.gz
        
    pip install cython
    cd ..
    cd ..
    git clone https://github.com/jrudar/TreeOrdination
    cd TreeOrdination
    python setup.py sdist
    cd dist
    pip install TreeOrdination-1.0.2.tar.gz
        
    pip install statsmodels==0.13.5
    pip install statannotations==0.5
    pip install matplotlib==3.5.2
    pip install seaborn==0.11.2

# Data Sources and References

Forbes JD, Chen C-Y, Knox NC, Marrie R-A, El-Gabalawy H, de Kievit T, et al. 
A comparative study of the gut microbiota in immune-mediated inflammatory 
diseases-does a common dysbiosis exist? Microbiome. 2018 Dec 13;6(1):221–221. 

Baxter NT, Ruffin MT, Rogers MAM, Schloss PD. Microbiota-based model improves 
the sensitivity of fecal immunochemical test for detecting colonic lesions. 
Genome Medicine. 2016 Apr 6;8(1):37

Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A, Knight R, et al. A 
Novel Sparse Compositional Technique Reveals Microbial Perturbations. mSystems. 
2019 Feb;4(1).

Porter, T. M., & Hajibabaei, M. (2022). MetaWorks: A flexible, scalable bioinformatic 
pipeline for high-throughput multi-marker biodiversity assessments. PLOS ONE, 
17(9), e0274260. doi: 10.1371/journal.pone.0274260

Wang, Q., Garrity, G. M., Tiedje, J. M., & Cole, J. R. (2007). Naive Bayesian 
Classifier for Rapid Assignment of rRNA Sequences into the New Bacterial Taxonomy. 
Applied and Environmental Microbiology, 73(16), 5261–5267. 
doi:10.1128/AEM.00062-07

Rudar, J., Porter, T.M., Wright, M., Golding G.B., Hajibabaei, M. LANDMark: an 
ensemble approach to the supervised selection of biomarkers in high-throughput 
sequencing data. BMC Bioinformatics 23, 110 (2022). 
https://doi.org/10.1186/s12859-022-04631-z
