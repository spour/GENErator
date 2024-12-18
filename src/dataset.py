import torch
from torch.utils.data import Dataset
import pandas as pd
import random
import torch.nn.functional as F

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores
        self.one_hot_sequences = [self.one_hot_encode(seq) for seq in self.sequences]
        self.scores = torch.tensor(self.scores, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.one_hot_sequences[idx]
        score = self.scores[idx]
        return seq, score

    def one_hot_encode(self, seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        seq_int = [mapping[nuc] for nuc in seq]
        seq_tensor = torch.tensor(seq_int, dtype=torch.long)
        one_hot = F.one_hot(seq_tensor, num_classes=4)  # shape: [seq_len, 4]
        return one_hot.float()


class BEDFileDataGenerator:
    def __init__(self, filepath, num_sequences=None, maxlen=1000):
        self.filepath = filepath
        self.num_sequences = num_sequences
        self.maxlen = maxlen
        self.data = self.load_data()

        # minmax norm 
        min_score = self.data['scores'].min()
        max_score = self.data['scores'].max()
        self.data['scores'] = (self.data['scores'] - min_score) / (max_score - min_score)

        # consensus is random for now
        self.consensus = "".join(random.choices(['A', 'C', 'G', 'T'], k=10))
    
    def load_data(self):
        """Loads sequence and score data from a BED file."""
        dataframe = pd.read_csv(self.filepath, sep='\t', header=None)
        # some sequences are so long they max the memory 
        dataframe = dataframe[dataframe[2] - dataframe[1] <= self.maxlen]
        # filter out n 
        last_col = dataframe.columns[-1]
        dataframe = dataframe[~dataframe[last_col].str.contains('N')]
        dataframe = dataframe[[6, last_col]].sample(n=self.num_sequences)
        dataframe.columns = ['scores', 'sequences']
        dataframe['sequences'] = dataframe['sequences'].str.upper()
        return dataframe
