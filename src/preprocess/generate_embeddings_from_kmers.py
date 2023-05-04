import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), 'DNABERT', 'src'))
sys.path.insert(1, '../')

is_python_3_6 = sys.version_info[0] == 3 and sys.version_info[1] == 6
if is_python_3_6:
    from transformers import BertModel, BertConfig, DNATokenizer

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2, parent_dir)

    
from config.config import DNABERTCONFIG


# Class for embedding generation for both dnabert and genslm
class Embeddings:
    def __init__(self, conf):
        self.config = conf
        self.model_type = "dnabert"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gets saved sequences
    def get_sequences(self, data_df):
        ens_ids = data_df['GeneID'].values.tolist()
        sequences = data_df['Seq'].values.tolist()
        scores = data_df['Avg Expr Lvl'].values.tolist()
        return ens_ids, sequences, scores

    # gets embeddings for dnabert or genslm
    def get_embeddings(self, ens_ids, sequences, scores):
        print("Getting embeddings for {} sequences".format(len(sequences)))
        # load dnabert model from pretrained_model 
        cfg = BertConfig.from_pretrained(DNABERTCONFIG())
        self.tokenizer = DNATokenizer.from_pretrained('dna{}'.format(self.config['kmer-size']))
        self.model = BertModel.from_pretrained(self.config['pretrained-model'], config=cfg).to(self.device)
        # set model to eval mode
        self.model.eval()
        self.max_length = self.config['max-sequence-length']
        return self.get_bert_embeddings(ens_ids, sequences, scores)

    def get_bert_embeddings(self, ens_ids, sequences, scores):
        embedding_list = []
        for i, sequence in enumerate(tqdm(sequences)):
            # append individual sequence embedding to embedding list
            embedding_list.append(self.individual_sequence_embedding(ens_ids[i], sequence, scores[i]))
            if (i+1) % 4000 == 0:
                print("Processed {} sequences".format(i+1))
                torch.save(embedding_list, self.config['embedding_dir'] + '/embeddings_{}.pt'.format(i+1))
            #if i == len(sequences) - 1:
            #    torch.save(embedding_list, self.config['embedding_dir'] + '/embeddings_{}.pt'.format(i+1))
        return embedding_list
    
    def individual_sequence_embedding(self, ens_id, sequence, score):
        total = 1
        for i in range(0,len(sequence),512):
            # get embedding for each 512 length chunk of sequence
            model_input = self.tokenizer.encode_plus(self.get_kmers_from_sequence(sequence), add_special_tokens=True, max_length=self.max_length)["input_ids"]
            model_input = torch.tensor(model_input, dtype=torch.long, device=self.device)
            model_input = model_input.unsqueeze(0)
            with torch.no_grad():
                last_hidden_states = self.model(model_input)
            if i == 0:
                embedding_value = last_hidden_states[1]
            else:
                embedding_value += last_hidden_states[1]
            total += 1
        return (ens_id, embedding_value/total, score)
    
    # join kmers into space separated string        
    def get_kmers_from_sequence(self, sequence, k=6):
        return ' '.join([sequence[i:i+k] for i in range(len(sequence)-k+1)])
