from torch.utils.data import DataLoader
import torch
import onnx
from ML.pruning_utils import prune_model
import os

def save_as_onnx_model(model: torch.nn.Module, output_path: str):
    '''
        Convert a TinyCnn /MNIST model to onnx
    '''
    model.eval()
    dummy = torch.randn(1,1,28,28) #dummy input
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        keep_initializers_as_inputs=False,
    )
    
    # Merge any external data into a single ONNX file
    if os.path.exists(output_path + ".data"):
        model_onnx = onnx.load(output_path, load_external_data=True)
        onnx.save_model(model_onnx, output_path, save_as_external_data=False)
        os.remove(output_path + ".data")

def quantize_export_pytorch(model: torch.nn.Module, model_name="tinycnn", prune_amount=0.3):
    
    ONNX_PATH = f"./ML/models/onnx/{model_name}.onnx"
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    
    # 1. Prune the model
    print(f'Pruning ({prune_amount*100}% of model model...')
    prune_model(model, prune_amount=prune_amount)
    
    # 2. Convert to ONNX
    print(f'Exporting PyTorch float model to ONNX...')
    save_as_onnx_model(model, ONNX_PATH)
    
if __name__ == "__main__":
    from ML.cnn_module import TinyCNN
    from ML.mnist_module import MNISTDataModule
    
    model = TinyCNN()
    dataset = MNISTDataModule()
    dataset.setup()
    val_loader = dataset.val_dataloader()
    quantize_export_pytorch(model, "dummy_model", 0.3)