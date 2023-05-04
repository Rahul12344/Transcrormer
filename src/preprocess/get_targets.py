import pandas as pd
import numpy as np

def get_targets_from_gtex_expression_values(c):
    print("Loading GTEx expression values...")
    df = pd.read_csv(c['files']['gtex_expression_values'])
    df['Avg Expr Lvl'] = np.log(df.mean(axis=1) + 1)
    
    target_df = df[['GeneID', 'Avg Expr Lvl']]
    
    start = df['GeneID'].str[:18]
    out_df = target_df.groupby(start).sum()
    
    return out_df