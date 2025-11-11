#include "ml.h"
#include "model_data.cc"  // embedded model (int8[])


// allocate the tensor arena
constexpr int kTensorArenaSize = 20 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// Instantiate a tflite model interpreter
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;


// Load the model from embedded C file
void init_model() {

    Serial.println("Loading embedded Model...");
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
    }

    // Resolve layers
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Load weights
    if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Failed to allocate tensors!");
    return;
    }

    // Assign input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Model initialized");
}

int8_t run_inference(const uint8_t* frame_data, size_t frame_len) {
  if (!interpreter || !input || !output) {
    Serial.println("ERROR: Model initialization failed!");
    return -1;
  }

  if (frame_len != input->bytes) {
    Serial.println("ERROR: invalid input size");
    return -1;
  }

  // Copy frame data to models input buffer
  memcpy(input->data.int8, frame_data, frame_len);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Error: Inference failed");
    return -1;
  }

  // Return produced output
  return output->data.int8[0];  
}
