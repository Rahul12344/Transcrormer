# Building a predictive model of gene expression values given only a transcript fasta file

## We build this model based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3852043/
From Zur and Tuller in BMC Bioinformatics, transcript features alone allow for accurate prediction of gene expression. While Hidden Markov models and covariance models are widely used in genomic analysis, large language models (LLMs) with transformer-based architecture have recently gained popularity in genomic problems.

## Setup
Install DNABERT from https://github.com/jerryji1993/DNABERT and follow setup guide. Install sklearn, pandas, matplotlib, and seaborn.

## Dimensionality Reduction
We explore various dimensionality reduction techniques such as Isomap and PCA to reduce the number of features.

## Data availability
All data can be found at https://drive.google.com/drive/folders/1h6pXP3DDKzwPltT5bEEY4hkMK_vEnewO?usp=sharing
