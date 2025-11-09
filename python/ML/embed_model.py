import torch
from pruning_utils import prune_model
from quant_utils import quantize_with_neural_compressor
from tflite_util import onnx_export, onnx_to_tflow, tflow_to_tflite, tflite_to_c_array

'''
    Prepare a SmallCNN and embed it onto the ESP32:
        -Prune the model with magnitude pruning
        -quantize the model to int8 using Intel Neural Compressor
        -Convert from PyTorch -> ONNX -> TFlite -> C array of int8s
'''

def model_to_embeddable(model, val_loader, model_name: str, prune_amount=0.3):
    
    TORCH_PATH = f'./ML/models/torch/{model_name}.torch'
    ONNX_PATH = f'./ML/models/onnx/{model_name}.onnx'
    TF_PATH = f'./ML/models/tf/{model_name}.tf'
    TFLITE_PATH = f'./ML/models/tflite/{model_name}.tflite'
    C_PATH = f'./ML/models/C/{model_name}.cc'
    
    
    #Step 1: Prune the model
    prune_model(model, prune_amount=prune_amount)

    #Step 2: Quantize the model
    quantize_with_neural_compressor(model, val_loader, outPath=TORCH_PATH)
    
    #Step 3: Convert quantized model to ONNX
    onnx_export(torch_path=TORCH_PATH, outputPath=ONNX_PATH)
    
    #Step 4: Convert to TensorFlow model
    onnx_to_tflow(ONNX_PATH, TF_PATH)
    
    #Step 5: Convert to Tflite model
    tflow_to_tflite(TF_PATH, TFLITE_PATH)
    
    #Step 5: C int8[] so it can be flashed as binary blob
    tflite_to_c_array(TFLITE_PATH, C_PATH)
    
    
if __name__ == "__main__":
    from cnn_module import TinyCNN
    from mnist_module import MNISTDataModule
    
    model = TinyCNN()
    val_loader = MNISTDataModule().val_dataloader()
    model_to_embeddable(model, val_loader, "dummy_model", 0.3)
    
