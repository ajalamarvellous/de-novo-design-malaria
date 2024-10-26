import os
import tqdm
import time

import mlflow

from typing import Union, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from livelossplot import PlotLosses

class Model(nn.Module):
    def __init__(self, input_shape, hidden_size, output_shape=1,): # drop_out=0.5):
        super(Model, self).__init__()
        self._input_shape = input_shape
        self._hidden_size = hidden_size
        self._output_layer = output_shape
        # self._dropout_size = drop_out

        # self._hidden_layer = nn.ModuleList([])
        for i, shape in enumerate(self._hidden_size):
            if i == 0:
                self._hidden_layer = nn.ModuleList([nn.Linear(self._input_shape, shape)])
            else:
                self._hidden_layer.append(nn.Linear(self._hidden_size[i-1], shape))

        # self._hidden_layer = nn.ModuleList(self._hidden_layer)
        self._output_layer = nn.Linear(self._hidden_size[-1], self._output_layer)
        # self._drop_out = nn.Dropout(self._dropout_size)

    def forward(self, data):
        input = data
        for layer in self._hidden_layer:
            input = layer(input)
            input = F.relu(input)
            # input = self._drop_out(input)

        out = self._output_layer(input)
        out = F.sigmoid(out)
        return out
    

class NNClassifier:
    def __init__(
        self,
        input_size: int,
        hidden_size: list,
        batch_size: int = 120,
        epochs: int=100,
        lr: float=0.1,
        metrics: dict = {},
        model_dir: str= "../models",
        track_mlflow: bool=True,
        class_weight: Union[List[float], None]=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_out = 0.5
        self.model = Model(self.input_size, self.hidden_size,) # drop_out=self.drop_out)

        if class_weight != None:
            self.criterion = torch.nn.BCELoss(weight=torch.Tensor([class_weight]))
        else:
            self.criterion = torch.nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.metrics = {}
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.metrics = metrics
        self.track_mlflow = track_mlflow
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.plotlosses = PlotLosses()
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.model_dir = model_dir

    def load_dataloader(self, X, y, batch_size):
        dataset = CustomDataset(X, y)
        return DataLoader(dataset, batch_size, shuffle=True)


    def predict(self, data, return_probas: bool = False):
        probas = self.model(data)
        predicted_labels = torch.where(probas > 0.5, 1, 0)
        if return_probas:
            return probas, predicted_labels
        else:
            return predicted_labels

    def fit(self, X, y, eval_every=5):
        self._train_loader = self.load_dataloader(X, y, batch_size=self.batch_size)
        for epoch in range(self.epochs): # tqdm.tqdm(, desc="Training"):
            self.model = self.model.train()
            loss = 0
            train_counter = 0
            start_time = time.time()
            print(f"epoch {epoch}/ {self.epochs}")
            for features, targets in self._train_loader:
                train_counter += 1
                # move data to device
                features = features.to(self.device)
                targets = targets.view(-1, 1).to(self.device)
                ### FORWARD AND BACK PROP
                probas = self.model(features)
                cost = self.criterion(probas, targets.type(torch.float))
                loss += cost.detach().numpy()
                self.optimiser.zero_grad()
                cost.backward()
                ### UPDATE MODEL PARAMETERS
                self.optimiser.step()
            loss = loss / train_counter
            self.model = self.model.eval()
            time_per_epoch = (time.time() - start_time) / 60
            if self.track_mlflow and ((epoch % eval_every == 0) or epoch == self.epochs - 1):
                mlflow.log_metric("train_loss", loss)
                mlflow.log_param("epoch_n", epoch)
                mlflow.log_param({"Time_per_epoch": time_per_epoch})
                self.save_model(epoch)
            
            self.plotlosses.update({"Train_loss": loss})
            self.plotlosses.update({"val_loss": self.metrics["Val_loss"]})
            print(f"epoch {epoch}/ {self.epochs}")

    
    def save_model(self, epoch):
        dir_ = self.model_dir + f"/{self.model._get_name()}_{epoch}.pth"
        torch.save(
            {
                "model_checkpoint": self.model.state_dict(),
                "optimiser_checkpoint": self.optimiser.state_dict(),
            },
            dir_,
        )
    
    def __str__(self):
        return "NNClassifier()"
    
    def get_params(self, deep=False):
        if deep:
            pass
        return {"input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "dropout": self.drop_out,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "lr": self.lr,
                "device": self.device.__str__(),
                "optimiser": "Adam", # Do not forget to alway change depending on what is used
                "loss_func": "BCELoss", # Do not forget to alway change depending on what is used
                }


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).type(torch.float)
        self.y = torch.from_numpy(y).type(torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
