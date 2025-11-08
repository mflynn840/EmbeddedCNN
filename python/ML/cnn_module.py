from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint


'''
    Simple CNN that is small enough to fit on an ESP32
    -32 channel conv, halving maxpool, 20 wide one layer network with leakyReLU activation functions
'''
class TinyCNN(nn.Module):
    def __init__(self, x):
        
        # x is (batch, 28, 28) 28x28 grayscale image batch
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=7,out_features=20)
        self.linear2 = nn.Linear(in_features=20,out_features=20)
        self.output = nn.Linear(in_features=20, out_features=10)
        self.leaky_relu = nn.LeakyReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.flatten(out)
        out = self.leaky_relu(self.linear1(out))
        out = self.leaky_relu(self.linear2(out))
        out = self.leaky_relu(self.output(out))
        return out
    

'''
    Train a TinyCNN using the following setup:
    - Use adamW and CosineAnnelaing LR to optimize
    - Use model checkpointing to pick the lowest validation loss model
    - Log information to weights and biases
'''
class TinyCnnModule(pl.LightningModule):
    def __init__(self):
        self.model = TinyCNN
        self.loss = nn.CrossEntropyLoss(self.model.parameters())
        self.accuracy = torchmetrics.Accuracy()
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss(preds, y)
        acc = self.accuracy(preds, y)
        self.log("train_loss_step", loss)
        self.log("train_acc_step", acc)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = self.loss(preds, y)
        self.log("val_loss_step", loss )
        return loss
        
        
    def configure_optimizers(self, optim_params, lrs_params):
        optimizer = AdamW(optim_params)
        lr_scheduler = CosineAnnealingLR(lrs_params)
        return {"optimizer" : optimizer, "lr_scheduler" : lr_scheduler}
        