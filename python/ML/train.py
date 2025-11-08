from pytorch_lightning import Trainer
from cnn_module import TinyCnnModule
from mnist_module import MNISTDataModule
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback



#launch a single training run
def train_mnist_tune(config, num_epochs=5, num_gpus=0):
    model = TinyCnnModule(lr=config["lr"])
    data = MNISTDataModule()
    
    tune_callback = TuneReportCallback({"val_loss" : "train_loss"}, on="validation_end")
    trainer = Trainer(
        max_epochs = num_epochs,
        accelerator = "gpu" if num_gpus else "cpu",
        devices = num_gpus if num_gpus else None,
        callbacks=[tune_callback]
    )
    
    trainer.fit(model, datamodule=data)
    
if __name__ == "__main__":
    search_space = {
        "lr" : tune.loguniform(1e-4, 1e-1),
        "batch_size" : tune.choice([32, 64])
    }
    
    tuner = tune.Tuner(
        tune.with_parameters(train_mnist_tune, num_epochs=5, num_gpus = 0),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=2 #trials per config
        ),
    )
    results = tuner.fit()
    print("Best conig:", results.get_best_result(metric="val_loss", mode="min0").config)
    