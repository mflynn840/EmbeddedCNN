#include <Arduino.h>
#include "get_frames.h"
#include "ml.h"

//Thread handles
TaskHandle_t inferenceTaskHandle = nullptr;
TaskHandle_t captureTaskHandle = nullptr;

//Mutex protects shared frame buffer
SemaphoreHandle_t frameMutex;

// Shared buffer for inference
uint8_t shared_frame[FRAME_SIZE];

// Thread 1: capture frames from serial or a camera
void capture_thread(void* param) {
    while(true){
        if(get_next_frame()) {
            //atomically copy the frame data to the shared buffer
            xSemaphoreTake(frameMutex, portMAX_DELAY);
            memcpy(shared_frame, frame_buffer, FRAME_SIZE);
            frame_ready = true;
            xSemaphoreGive(frameMutex);
        }
    }
}

// Thread 2: run inference on captured frames
void inference_thread(void* param) {
    while(true) {
        bool ready = false;

        //atomically consume the frame
        xSemaphoreTake(frameMutex, portMAX_DELAY);
        ready = frame_ready;
        if (frame_ready) frame_ready = false;
        xSemaphoreGive(frameMutex);

        //write inference result to serial
        if (ready) {
            int8_t result = run_inference(shared_frame, sizeof(shared_frame));
            Serial.print("Inference result: ");
            Serial.println(result);
        }

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

    //create sychronization semaphore
    frameMutex = xSemaphoreCreateMutex();

    //Start the two demo threads (model inference and frame capture)
    xTaskCreatePinnedToCore(serial_thread_task, "SerialTask", 4096, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(capture_thread, "CaptureThread", 4096, NULL, 1, &captureTaskHandle, 0);
    xTaskCreatePinnedToCore(inference_thread, "InferenceThread", 8192, NULL, 1, &inferenceTaskHandle, 1);

}

void loop() {
    vTaskDelay(1000 / portTICK_PERIOD_MS);
}


