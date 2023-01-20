# Unsupervised-Decision-Trees

Data to support our paper titled: "Decision Tree Ensembles Utilizing Multivariate Splits Are Effective at Investigating Beta-Diversity in Medically Relevant 16S Amplicon Sequencing Data"
https://www.biorxiv.org/content/10.1101/2022.03.31.486647v2

# Create the Environment
  conda create -n TreeOrdFinal python=3.10

  # Activate the Environment
  conda activate TreeOrdFinal

  # Install LANDMark
  git clone https://github.com/jrudar/LANDMark
  cd LANDMark
  python setup.py sdist
  cd dist
  pip install LANDMark-1.2.0.tar.gz

  # Install TreeOrdination
  pip install cython
  cd ..
  cd ..
  git clone https://github.com/jrudar/TreeOrdination
  cd TreeOrdination
  python setup.py sdist
  cd dist
  pip install TreeOrdination-1.0.2.tar.gz


  # Install Other Packages Which Are Needed
  pip install statsmodels==0.13.5
  pip install statannotations==0.5
  pip install matplotlib==3.5.2
  pip install seaborn==0.11.2

# Data Sources

Forbes JD, Chen C-Y, Knox NC, Marrie R-A, El-Gabalawy H, de Kievit T, et al. 
A comparative study of the gut microbiota in immune-mediated inflammatory 
diseases-does a common dysbiosis exist? Microbiome. 2018 Dec 13;6(1):221–221. 

Baxter NT, Ruffin MT, Rogers MAM, Schloss PD. Microbiota-based model improves 
the sensitivity of fecal immunochemical test for detecting colonic lesions. 
Genome Medicine. 2016 Apr 6;8(1):37

Martino C, Morton JT, Marotz CA, Thompson LR, Tripathi A, Knight R, et al. A 
Novel Sparse Compositional Technique Reveals Microbial Perturbations. mSystems. 
2019 Feb;4(1).
