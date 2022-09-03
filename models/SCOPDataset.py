import sys
sys.path.append("../scop_classification_by_ESM")

import torch
from torch.utils.data import Dataset
import pandas as pd
from Bio import SeqIO
from sklearn.utils.class_weight import compute_class_weight

def generate_class_dict(all_data_file_path, task="SF"):
    # generating class dictionary
    df = pd.read_csv(all_data_file_path)
    x = df[task].unique().tolist()
    class_dict = {j:i for i,j in enumerate(x)}
    n_classes = len(class_dict)
    print(f"n_classes: {n_classes}")
    return class_dict, n_classes

def compute_class_weights(train_data_file_path, task="SF"):
    # computing class weights from the train data
    train_df = pd.read_csv(train_data_file_path)
    class_weights = compute_class_weight("balanced", classes=train_df[task].unique(), y=train_df[task].to_numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)#, device=device)
    # print(train_df[task].value_counts(sort=False))
    # print(f"class_weights: {class_weights}")
    return class_weights



class X(Dataset):
    def __init__(self, inp_file, esm_batch_converter, class_dict, task="SF", max_len=512) -> None:
        super(X, self).__init__()

        self.esm_batch_converter = esm_batch_converter
        self.df = pd.read_csv(inp_file)
        self.class_dict = class_dict
        self.task = task
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def cut_and_padd(self, seq_tokens):
        x = torch.tensor([1]).repeat(self.max_len) # padding with 1's
        seq_tokens = seq_tokens[:self.max_len]
        x[:seq_tokens.shape[0]] = seq_tokens
        return x


    def __getitem__(self, index):
        row = self.df.loc[index]
        pdb_id, chain_and_region = row["FA-PDBID"].lower(), row["FA-PDBREG"].split(":")
        chain_id, region = chain_and_region[0], chain_and_region[1]

        record = next(SeqIO.parse("data/fastas/"+pdb_id+chain_id+region+".fasta", "fasta"))
        id, seq = pdb_id+chain_id+region, str(record.seq)


        batch_labels, batch_strs, seq_tokens = self.esm_batch_converter([(id, seq)])
        seq_tokens = self.cut_and_padd(seq_tokens.squeeze(0))
        

        # making ground-truth class tensor
        class_id = self.class_dict[self.df.loc[index, self.task]]
        label = torch.tensor(class_id, dtype=torch.long)
        
        # print(seq_tokens.shape, label, len(seq))
        return seq_tokens, label, len(seq)