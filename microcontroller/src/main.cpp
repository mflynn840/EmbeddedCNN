#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"


const int FRAME_SIZE = 28*28;
uint8_t frame[FRAME_SIZE];

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Serial.println("ESP32 ready to receive frames");
    // TODO: Load TFLite model
}

void loop() {
    if (Serial.available() >= FRAME_SIZE) {
        Serial.readBytes(frame, FRAME_SIZE);
        Serial.println("Frame received!");
        // TODO: Run inference here
    }
}
