from torch import nn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torchmetrics

from ML.mnist_module import MNISTDataModule


'''
    Simple CNN that is small enough to fit on an ESP32
    -32 channel conv, halving maxpool, 20 wide one layer network with leakyReLU activation functions
'''
class TinyCNN(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1568,out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20,out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=10)
        )
           
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    @classmethod
    def from_state_dict(cls, path: str, map_location="cpu"):
        """Load a TinyCNN from a state_dict"""
        model = cls()
        state_dict = torch.load(path, map_location=map_location)
        model.load_state_dict(state_dict)
        return model
    

'''
    Train a TinyCNN using the following setup:
    - Use adamW and CosineAnnelaing LR to optimize
    - Use model checkpointing to pick the lowest validation loss model
    - Log information to weights and biases
'''
class TinyCnnModule(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = TinyCNN()
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.lr = lr        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        self.log("train_loss_step", loss, prog_bar=True)
        self.log("train_acc_step", acc)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss_step", loss )
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]
        
if __name__ == "__main__":
    model = TinyCnnModule(lr=.001)
    data = MNISTDataModule()
    
    trainer = pl.Trainer(
        max_epochs = 1,
        accelerator = "cpu",
        devices = 1,
    )
    
    trainer.fit(model, datamodule=data)
    