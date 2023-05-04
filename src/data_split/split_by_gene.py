from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

def data_set_split(c, genes):
    print('Splitting data into training and testing sets...')
    gene_dummy_labels=np.zeros(len(genes))
    train_genes, test_genes, _, _ = train_test_split(genes, gene_dummy_labels, test_size=c['test-size'], random_state=c['random-state'])
    return train_genes, test_genes

def build_train_test_sets(data_df, train_genes, test_genes):
    print('Building training and testing sets...')
    train_df = data_df[data_df['GeneID'].isin(train_genes)]
    test_df = data_df[data_df['GeneID'].isin(test_genes)]
            
    return train_df, test_df