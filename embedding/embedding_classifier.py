import os
from typing import Dict, List

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import Dataset

class EmbeddingClassifier(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        self.embed_model = SentenceTransformer("all-mpnet-base-v2")
        self.embed_model.requires_grad_(False)
        self.classifier = nn.Linear(768, 3)
        if model_path is not None:
            if os.path.exists(model_path):
                print(f"Loading {model_path}")
                self.classifier.load_state_dict(torch.load(model_path))
            else:
                print("No trained model found.")
        else:
            print("No model provided, use a random initialized model")
        self.map_name = ["coding", "math", "none"]
        self.model = f"Embedding Classifier {model_path}"

    def forward(self, prompts: List[str]) -> bool:
        embeddings = self.embed_model.encode(prompts)
        embeddings_tensor = torch.vstack([torch.Tensor(embedding) for embedding in embeddings])
        return self.classifier(embeddings_tensor)

    def classify_prompt(self, prompt: str) -> str:
        _, idx = torch.max(self.forward([prompt])[0], 0)
        pred_class = self.map_name[idx.item()]
        return pred_class

    def train(self, dataloader, optimizer, epochs, loss_fn=torch.nn.CrossEntropyLoss()):
        for epoch in range(epochs):
            for batch in dataloader:
                prompts = list(batch["Prompt"][0])
                labels = batch["Label"][0].to(torch.long)
                optimizer.zero_grad()
                output = self(prompts)
                assert output.shape[0] == labels.shape[0] and output.shape[1] == 3
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                print(loss.item())
            torch.save(self.classifier.state_dict(), f"embedding_model_{epoch + 1}.pt")


class DatasetFromPandas(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.MAP = {"Programming": 0, "Math": 1, "Other": 2}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]
        X = self.dataframe.iloc[idx, :-1].values.tolist()
        Y = self.dataframe.iloc[idx, -1].tolist()
        X = [x[0] for x in X]
        Y = [self.MAP[y] for y in Y]
        return {"Prompt": X, "Label": Y}

if __name__ == "__main__":
    dataset = DatasetFromPandas(pd.read_csv("merged_data.csv"))
    dataloaders = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    classifier = EmbeddingClassifier()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    epochs = 3
    classifier.train(dataloaders, optimizer, epochs=epochs)

    str = "Write a program that takes a list of numbers and prints each number in the list that is even."
    print(classifier.forward([str]))
    print(classifier.classify_prompt(str))
