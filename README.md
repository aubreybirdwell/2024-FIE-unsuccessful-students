# Overview

This repository contains supplementary materials for the following conference paper:

Valdemar Švábenský, Kristián Tkáčik, Aubrey Birdwell, Richard Weiss, Ryan S. Baker, Pavel Čeleda, Jan Vykopal, Jens Mache, and Ankur Chattopadhyay.\
**Detecting Unsuccessful Students in Cybersecurity Exercises in Two Different Learning Environments**\
In Proceedings of the 54th Frontiers in Education Conference (FIE 2024).

# How to cite

If you use or build upon the materials, please use the BibTeX entry below to cite the original paper (not only this web link).

```bibtex
@inproceedings{Svabensky2024detecting,
    author    = {\v{S}v\'{a}bensk\'{y}, Valdemar and Tk\'{a}\v{c}ik, Kristi\'{a}n and Birdwell, Aubrey and Weiss, Richard and Baker, Ryan S. and \v{C}eleda, Pavel and Vykopal, Jan and Mache, Jens and Chattopadhyay, Ankur},
    title     = {{Detecting Unsuccessful Students in Cybersecurity Exercises in Two Different Learning Environments}},
    booktitle = {Proceedings of the 54th Frontiers in Education Conference},
    series    = {FIE '24},
    publisher = {IEEE},
    address   = {New York, NY, USA},
    year      = {2024},
    numpages  = {9},
}
```

# Contents of the repository

* Dataset (directory ``dataset``)
* Code (directory ``src``)
* Results (directory ``results``)

Each part is described below.

## Dataset (see Section III-C of the paper)

Dataset used for feature extraction and model training.

If you use the data, please cite them according to the instructions on the link.

*Note: Some data fields have been redacted because they could reveal the solution to training tasks.*

## Code (see Section III-G and III-H of the paper)

The scripts were tested on a Linux Mint 19 and 20 systems with Python versions 3.9 and 3.10.

The included `requirements.txt` file can be used to install the required Python packages:

```commandline
pip install -r requirements.txt
```

### feature_extraction.py

The purpose of this script is to extract features from the dataset. The extracted features are used in `classification.py` to train various classifiers.

This script can be executed without any arguments. It parses the `../dataset` directory, prints statistics about the parsed dataset, and creates two Kendall correlation matrices:

* Correlation matrix in `corr_matrix_all.pdf` with all extracted features,
* Correlation matrix in `corr_matrix_best.pdf` with a subset of features that have the strongest correlation with the target variable (named *struggled* here and *unsuccessful* in the paper).

The correlation matrices are exported to `./results` directory. If the directory does not exist, the script creates it.

### classification.py

Run this script without any arguments. It parses the `../dataset` directory, prints statistics about the dataset, extracts features from it, and uses them to train and evaluate the following 8 classifiers:

* logistic regression
* naive Bayes
* support vector machine with linear kernel
* support vector machine with RBF kernel
* nearest neighbors
* decision tree
* random forest
* XGBoost

The script exports two files for each classifier:

* Result table in `<classifier name>_res.txt`, which includes evaluation metric values, best hyperparameter configuration, and selected features for each model trained in the 10-fold cross-validation used for classifier evaluation. The table also includes the average value of each evaluation metric.
* Figure in `<classifier name>_cm.pdf`, which contains confusion matrices of the models trained in the 10-fold cross-validation used for classifier evaluation.

The script also exports figures with bar charts displaying feature importance weights for 5 of the 8 models: logistic regression, linear SVM, decision tree, random forest, and XGBoost. The files containing these figures are named `<classifier name>_importance.pdf`.

If the target directory does not exist, the script creates it. The script finishes execution in approximately **10 minutes** on an Intel Core i7 system with 12 cores and 16 GB of RAM. All exported files are stored in the `./results` directory.

## Results (see Section IV-A of the paper)

The complete results for both the full dataset and half of the dataset.

The `figures` directory contains all figures that were exported using the scripts detailed above.

The `tables` directory contains text files created by the `classification.py` script. Each file corresponds to one of the trained classifiers and contains a table with results of the 10-fold cross-validation method used for classifier evaluation.

## License

MIT (see the `LICENSE.md` file).\
Please cite the paper according to the instructions above.
