import torch
from src.data import BEDFileDataGenerator
from src.models import MultiTaskDiffusionModel
from src.diffusion import DiscreteDiffusion

# like a bedtools -bedOut structure
data_generator = BEDFileDataGenerator(
    filepath='/project/6000369/spour98/ledidi/data/ENCFF852AAQ.bedout', 
    num_sequences=15000, 
    maxlen=512
)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn.functional as F
from src.data import BEDFileDataGenerator, DNASequenceDataset
from src.models import MultiTaskDiffusionModel
from src.diffusion import DiscreteDiffusion
from src.training import train_multi_task

# Setup data generator
data_generator = BEDFileDataGenerator(
    filepath='/project/6000369/spour98/ledidi/data/ENCFF852AAQ.bedout', 
    num_sequences=15000, 
    maxlen=512
)

dataframe = data_generator.data
sequences = dataframe['sequences'].values
scores = dataframe['scores'].values

class DNASequenceDataLoader(Dataset):
    def __init__(self, sequences, scores):
        self.dataset = DNASequenceDataset(sequences, scores)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq, score = self.dataset[idx]
        mask = torch.ones(seq.size(0), dtype=torch.bool)  
        return seq, score, mask

batch_size = 32
dataset = DNASequenceDataLoader(sequences, scores)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_classes = 4
hidden_size = 128
T = 50
epochs = 10  

model = MultiTaskDiffusionModel(num_classes=num_classes, hidden_size=hidden_size, num_layers=6, num_heads=8)
diffusion = DiscreteDiffusion(model=model, timesteps=T, mode="multi")

learning_rate = 1e-4
optimizer = Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


if __name__ == "__main__":
    print("starting training")
    train_multi_task(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        weight=0.5,  #
    )
    print("DONE!")
