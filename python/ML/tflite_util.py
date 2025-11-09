import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def onnx_export(torchPath:str, outputFileName: str):
    model = torch.load(torchPath)
    
    dummy = torch.randn(1,1,28,28) #dummy input
    torch.onnx.export(
        model,
        dummy,
        f'{outputFileName}.onnx',
        opset_version=13,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    )


def onnx_to_tflow(onnx_path, output_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(output_path)
    

def tflow_to_tflite(tflow_path:str, tflite_path:str):
    '''
        Convert a quantized (int8) tensorFlow model to TFLITE
    '''
    converter = tf.lite.TFLiteConverter.from_saved_model(tflow_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
def tflite_to_c_array(tflite_path: str, output_path: str, var_name: str = "tflite_model"):
    """
    Convert a .tflite file into a C array suitable for embedding in firmware.
    
    Args:
        tflite_path (str): Path to the .tflite model file.
        var_name (str): Name of the C array variable.
        output_path (str): Output .cc file path (defaults to same basename + .cc).
    """
    import os
    
    if output_path is None:
        output_path = os.path.splitext(tflite_path)[0] + "_data.cc"

    with open(tflite_path, "rb") as f:
        data = f.read()

    hex_bytes = ", ".join(f"0x{b:02x}" for b in data)
    array_len = len(data)

    c_code = f"""#include <cstddef>
                #include <cstdint>

                const unsigned char {var_name}[] = {{
                                        {hex_bytes}
                                        }};
                const unsigned int {var_name}_len = {array_len};
                """

    with open(output_path, "w") as f:
        f.write(c_code)

    print(f"Generated {output_path} with {array_len} bytes as {var_name}[]")

        
    