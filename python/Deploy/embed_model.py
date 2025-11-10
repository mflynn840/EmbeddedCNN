from python.ML.pruning_utils import prune_model
from Deploy.embed_model_utils import onnx_export, onnx_to_tflow, tflow_to_quant_tflite, tflite_to_c_array
import os
'''
    Prepare a SmallCNN for embedding:
        -Prune the model with magnitude pruning
        -quantize the model to int8 using Intel Neural Compressor
        -Convert from PyTorch -> ONNX -> TFlite -> C array of int8s
'''

def embed_onnx_model(onnx_model_path:str,val_loader: DataLoader, model_name="tinycnn", prune_amount=0.3):
    
    #make paths for all intermediate models
    
    TF_PATH = f"./ML/models/tf/{model_name}"
    TFLITE_PATH = f"./ML/models/tflite/{model_name}.tflite"
    C_PATH = f"./ML/models/C/{model_name}.cc"
    os.makedirs(os.path.dirname(TF_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(C_PATH), exist_ok=True)

    representative_data_gen = representative_data_gen_from_loader(val_loader)


    # 3. Convert to TF
    print(f'Converting ONNX model to tensorflow float model...')
    onnx_to_tflow(ONNX_PATH, TF_PATH)

    # 4. Quantize in TF → TFLite
    print(f'Converting TFlow float model to quantized int8 tflowlite model...')
    tflow_to_quant_tflite(TF_PATH, TFLITE_PATH, representative_data_gen)

    # 5. Convert to C array
    print(f'Convert to an embeddable C array of int8')
    tflite_to_c_array(TFLITE_PATH, C_PATH)
                      
                      
if __name__ == "__main__":
    from ML.cnn_module import TinyCNN
    from ML.mnist_module import MNISTDataModule
    
    model = TinyCNN()
    dataset = MNISTDataModule()
    dataset.setup()
    val_loader = dataset.val_dataloader()
    export_as_embeddable(model, val_loader, "dummy_model", 0.3)
    
