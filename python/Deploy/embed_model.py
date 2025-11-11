from Deploy.embed_model_utils import onnx_to_tflow, tflow_to_quant_tflite, tflite_to_c_array
import numpy as np


import os
'''
    Prepare a SmallCNN for embedding:
        -Prune the model with magnitude pruning
        -quantize the model to int8 using Intel Neural Compressor
        -Convert from PyTorch -> ONNX -> TFlite -> C array of int8s
'''

def embed_onnx_model(onnx_model_path:str, X_array: np.ndarray, model_name="tinycnn", prune_amount=0.3):
    
    #make paths for all intermediate models
    TF_PATH = f"./ML/models/tf/{model_name}"
    TFLITE_PATH = f"./ML/models/tflite/{model_name}.tflite"
    C_PATH = f"./ML/models/C/{model_name}.cc"
    os.makedirs(os.path.dirname(TF_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(TFLITE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(C_PATH), exist_ok=True)

    representative_data_gen = representative_data_gen_from_np(X_array)

    # 3. Convert ONNX model to TF model
    print(f'Converting ONNX model to tensorflow float model...')
    onnx_to_tflow(onnx_model_path, TF_PATH)

    # 4. Quantize in TF → TFLite
    print(f'Converting TFlow float model to quantized int8 tflowlite model...')
    tflow_to_quant_tflite(TF_PATH, TFLITE_PATH, representative_data_gen)

    # 5. Convert to C array
    print(f'Convert to an embeddable C array of int8')
    tflite_to_c_array(TFLITE_PATH, C_PATH)


'''Return an iterable with a representative dataset for model quantization'''
def representative_data_gen_from_np(X_array: np.ndarray):    
    def representative_data_gen():
        for input_value in X_array:
            input_value = np.expand_dims(input_value, axis=0).astype(np.float32)
            yield [input_value]
    return representative_data_gen
        
      
if __name__ == "__main__":
    ONNX_PATH = f"./ML/models/onnx/dummy_model.onnx"
    X_array = np.load("./Deploy/representative_data.npy")
    embed_onnx_model(ONNX_PATH, X_array, "model", 0.3)
    
