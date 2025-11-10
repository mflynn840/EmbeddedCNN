from collections.abc import Iterable
import torch
import onnx
import tensorflow as tf
import os
from onnx import numpy_helper 
import numpy as np
import onnx
import numpy as np
from onnx import numpy_helper
from onnxsim import simplify
from onnx_tf.backend import prepare


def onnx_export(model: torch.nn.Module, output_path: str):
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

        

def onnx_to_tflow(onnx_path, output_path):
    '''
        Convert an Onnx TinyCNN to a tensorflow model
    '''
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    
    # Simiplify the model
    model, check = simplify(model)
    if not check:
        raise RuntimeError("ONNX simplifier failed.")
    
    # Convert to TensorFlow
    tf_rep = prepare(model)
    tf_rep.export_graph(output_tf_path)
    

def tflow_to_quant_tflite(tflow_path:str, tflite_path:str, representative_data_gen: Iterable):
    '''
        Convert a quantized (int8) tensorFlow model to TFLITE
    '''
    converter = tf.lite.TFLiteConverter.from_saved_model(tflow_path)
    converter.representative_dataset = representative_data_gen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

     
def tflite_to_c_array(tflite_path: str, output_path: str, var_name="tflite_model"):
    with open(tflite_path, "rb") as f:
        data = f.read()
    hex_bytes = ", ".join(f"0x{b:02x}" for b in data)
    array_len = len(data)

    c_code = f"""#include <cstdint>
                const unsigned char {var_name}[] = {{
                {hex_bytes}
                }};
                const unsigned int {var_name}_len = {array_len};
            """
    with open(output_path, "w") as f:
        f.write(c_code)
    print(f"[INFO] Generated C array at {output_path} ({array_len} bytes)")
    
    
