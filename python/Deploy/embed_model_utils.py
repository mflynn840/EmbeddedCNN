from collections.abc import Iterable
import tensorflow as tf
import onnx
from onnxsim import simplify
from onnx_tf.backend import prepare
import os



        

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
    
    
