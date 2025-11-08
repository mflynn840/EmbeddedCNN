



def run_kfold(model_class, datamodule, k=5, trainer_kwargs={}):
    val_accs = []
    for fold in range(k):
        model = model_class()
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, datamodule=datamodule, ckpt_path=None)
        metrics = trainer.validate(model, datamodule=datamodule)
        val_accs.append(metrics[0]['val_acc'])
    return val_accs