
import torch
from neural_compressor.quantization import fit 
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion


def quantize_with_neural_compressor(model: torch.nn.Module, val_dataloader, outPath:str):
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion,
        accuracy_criterion=accuracy_criterion
    )
    
    q_model = fit(model = model.model, conf=conf,calib_dataloader=val_dataloader)
    q_model.save(outPath)
  


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
    
'''Quantization aware training utilities'''

