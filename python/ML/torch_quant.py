import torch

#TODO: implement static quantization yourself based on the pytorch tutorial
class ObservedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, 
                 out_features: int, 
                 activation_observer: torch.nn.Module,
                 weight_observer: torch.nn.Module,
                 bias: bool = True,
                 device = None,
                 dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

import numpy as np

def representative_data_gen_from_loader(val_loader, num_batches=100):
    for i, (images, _) in enumerate(val_loader):
        if i >= num_batches:
            break
        # Convert torch.Tensor → numpy array, match TFLite input shape [N, H, W, C]
        images = images.permute(0, 2, 3, 1).numpy().astype(np.float32)
        yield [images]


'''
def quantize_with_neural_compressor(model: torch.nn.Module, val_dataloader, outPath:str):
    from neural_compressor.quantization import fit 
    from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion,
        accuracy_criterion=accuracy_criterion
    )
    
    q_model = fit(model = model, conf=conf,calib_dataloader=val_dataloader)
    q_model.save(outPath)
'''

