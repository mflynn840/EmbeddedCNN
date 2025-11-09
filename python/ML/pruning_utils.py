import torch
import torch.nn.utils.prune as prune


def global_magintude_prune(model, amount=0.3):
    '''
        Prune a certain percentage of weights based on magnitude
    '''
    prunable_params = []
    
    #collect all parameters from the models conv and linear layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prunable_params.append((module, "weight"))
    
    prune.global_unstructured(
        prunable_params,
        pruning_method = prune.L1Unstructured,
        amount = amount
    )
    return prunable_params

def apply_pruning(prunable_params):
    '''Finalize pruning by removing reparameterization'''
    for module, name in prunable_params:
        prune.remove(module, name)
        
   
def prune_model(model:torch.nn.Module, amount=0.3):
    pruned = global_magintude_prune(model, amount=amount)
    apply_pruning(pruned)
    
#usage     
if __name__ == "__main__":
    from cnn_module import TinyCnnModule
    model = TinyCnnModule(lr=1e-3)
    pruned = global_magintude_prune(model, amount=0.3)
    #optionally fine tune for a few epochs to regain acc
    #make pruning permanent:
    apply_pruning(pruned)
            