import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

def parse_transcript(c):
    print('Parsing transcript fasta file...')
    x = []
    with open(c['files']['transcript_fasta']) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            gene_name = record.description.split(' | ')[1][5:]
            sequences = str(record.seq)
            x.append({"GeneID": gene_name, "Seq": sequences})
    
    data_df = pd.DataFrame(x)
    return data_df

def add_target_gtex_scores_to_transcripts(seq_df, target_df): 
    print('Adding target GTEx scores to transcripts...')
    data = []   
    ens_ids = seq_df['GeneID'].values.tolist()
    for i, ens_id in tqdm(enumerate(ens_ids)):
        try:
            data.append([ens_id, seq_df['Seq'].iloc[i], target_df.loc[ens_id, 'Avg Expr Lvl']])
        except:
            data.append([ens_id, seq_df['Seq'].iloc[i], 0])
            
    data_df = pd.DataFrame(data, columns=['GeneID', 'Seq', 'Avg Expr Lvl'])
    return data_df
    