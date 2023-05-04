import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# import dimensionality reduction functions for visualization
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding

# get parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add parent directory to path
sys.path.insert(0, parent_dir)

# import config
from config.config import config

# import preprocessing functions
from preprocess.get_targets import get_targets_from_gtex_expression_values
from preprocess.parse_transcript_and_generate_kmers import parse_transcript, add_target_gtex_scores_to_transcripts

# import data splitting functions
from data_split.split_by_gene import data_set_split, build_train_test_sets

# import embedding functions
from preprocess.generate_embeddings_from_kmers import Embeddings

# import model training functions
from sklearn_models.train_model_pipeline import train_sklearn_model_and_return_pipeline, tune_sklearn_pipeline, predict_from_sklearn_pipeline 

# import model evaluation functions
from utils.model_params import params, dimensionality_reduction_functions

if __name__ == '__main__':
    

    # get config file
    c = config()
    
    # initialize embeddings object
    embedder = Embeddings(c)
    
    # build target values dataframe
    target_df = get_targets_from_gtex_expression_values(c)
    
    # build sequence dataframe
    seq_df = parse_transcript(c)
    
    # assign target values to sequences
    data_df = add_target_gtex_scores_to_transcripts(seq_df, target_df)
    
    # randomly sample subset for time efficiency
    data_df = data_df.sample(frac=c['subset-size'])
    
    # plot histogram of target values
    sns.set(color_codes=True)
    sns.distplot(data_df['Avg Expr Lvl'], color='purple', kde=False, bins=100)
    plt.xlabel('Gene Expression Level')
    plt.ylabel('Number of Genes in Expression Level Range')
    plt.title('Gene Expression Level Distribution')
    plt.savefig('histogram.png')
    plt.clf()
    
    # split data into train and test sets
    train_genes, test_genes = data_set_split(c, data_df['GeneID'].unique())
    train_df, test_df = build_train_test_sets(data_df, train_genes, test_genes)
    
    # get sequences and embeddings for train and test sets
    ens_ids, seqs, scores = embedder.get_sequences(train_df)
    training_embeddings = embedder.get_embeddings(ens_ids, seqs, scores)
    X_train, y_train = torch.stack([embedding[1] for embedding in training_embeddings]), torch.FloatTensor([embedding[2] for embedding in training_embeddings])
    
    ens_ids, seqs, scores = embedder.get_sequences(test_df)
    testing_embeddings = embedder.get_embeddings(ens_ids, seqs, scores)
    X_test, y_test = torch.stack([embedding[1] for embedding in testing_embeddings]), torch.FloatTensor([embedding[2] for embedding in testing_embeddings])
    
    # convert embeddings to numpy arrays
    X_train = X_train.cpu().detach().numpy()
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))
    y_train = y_train.cpu().detach().numpy()
    
    X_test = X_test.cpu().detach().numpy()
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    y_test = y_test.cpu().detach().numpy()
    
    """
    # visualize PCA of training data in 2D
    dimensionality_reduction = PCA(n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('PCA of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('pca_2d.png')
    plt.clf()
    
    # visualize Isomap of training data in 2D
    dimensionality_reduction = Isomap(n_neighbors=5, n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('Isomap of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('isomap_2d.png')
    plt.clf()
    
    # visualize MDS of training data in 2D
    dimensionality_reduction = MDS(n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.fit_transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('MDS of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('mds_2d.png')
    plt.clf()
    
    # visualize tSNE of training data in 2D
    dimensionality_reduction = TSNE(n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.fit_transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('tSNE of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('tsne_2d.png')
    plt.clf()
    
    # visualize LLE of training data in 2D
    dimensionality_reduction = LocallyLinearEmbedding(n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.fit_transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('LLE of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('lle_2d.png')
    plt.clf()
    
    # visualize SE of training data in 2D
    dimensionality_reduction = SpectralEmbedding(n_components=2)
    dimensionality_reduction.fit(X_train)
    r2_train_data = dimensionality_reduction.fit_transform(X_train)
    
    sns.set(color_codes=True) 
    df = pd.DataFrame({'x': r2_train_data[:,0], 'y': r2_train_data[:,1], 'label': y_train})
    sns.scatterplot(data=df, x='x', y='y', hue='label')
    plt.title('SE of Training Data in 2D')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig('se_2d.png')
    plt.clf() """
    
    # train model
    print("Training model and tuning over hyperparameters...")
    rfr = RandomForestRegressor()
    pipeline = train_sklearn_model_and_return_pipeline(scaler=StandardScaler(), dimensionality_reduction=dimensionality_reduction_functions[c['dimensionality-reduction']], model=rfr)
    optimal_pipeline = tune_sklearn_pipeline(pipeline, params, X_train, y_train)
    prediction, score = predict_from_sklearn_pipeline(optimal_pipeline, X_test, y_test)
    predictions_and_actuals = pd.DataFrame({'Gene Transcript Sample Index': list(range(len(prediction))), 'Predicted Expression': prediction, 'Actual Expression': y_test})
    predictions_and_actuals.to_csv('predictions.csv')
    
    sns.set(color_codes=True)
    sns.scatterplot(data=predictions_and_actuals, x='Gene Transcript Sample Index', y='Predicted Expression', color='red')
    sns.scatterplot(data=predictions_and_actuals, x='Gene Transcript Sample Index', y='Actual Expression', color='purple')
    plt.legend(['Predicted Expression', 'Actual Expression'])
    plt.xlabel('Gene Transcript Sample Index')
    plt.ylabel('Gene Expression Level')
    plt.title('Predicted vs. Actual Gene Expression Levels')
    
    plt.savefig('predictions.png')
    plt.clf()
    