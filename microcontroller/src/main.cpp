#include <Arduino.h>
#include "get_frams.h"
#include "ml.h"

TaskHandle_t inferenceTaskHandle;
TaskHandle_t getFramesTaskHandle;

// Define the model inference thread
void inference_thread(void* param){
    while(true){
        //get the next frame using the get_frame package
        if(frame_ready){
            frame_ready = false;
            int8_t result = run_inference(frame_buffer, sizeof(frame_buffer));
            Serial.print("Inference result: ");
            Serial.println(result);

        }
        //avoid starving other tasks
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}

void setup() {

    //connect to serial
    Serial.begin(115200);
    while (!Serial);

    Serial.println("\n====ESP 32 TFlite demo =======");

    //Load the embedded tflite flatbuffer encoded model
    init_model();

    //Start the two demo threads (model inference and frame capture)
    xTaskCreatePinnedToCore(serial_thread_task, "SerialThread", 4096, NULL, 1, &serialTaskHandle, 0);
    xTaskCreatePinnedToCore(inference_thread_task, "InferenceThread", 8192, NULL, 1, &inferenceTaskHandle, 1);

}

void loop() {
    vtaskDelay(1000 / portTICK_PERIOD_MS);
}


