#include "get_frames.h"
#include <Arduino.h>

uint8_t frame_buffer[FRAME_SIZE];      
extern SemaphoreHandle_t frameReadySemaphore;  // from main.cpp


// Read bytes from serial into frame_buffer
void serial_thread_task(void* param) {
  static uint16_t index = 0; // current write position in frame_buffer

  while (true) {

    // attomically pull serial input into buffer
    while (Serial.available() > 0) {
      uint8_t byte = Serial.read();
      frame_buffer[index++] = byte;

      // Check if a full frame is received
      if (index >= FRAME_SIZE) {
        index = 0;           // reset for next frame
        Serial.println("Frame received");
        xSemaphoreGive(frameReadySemaphore);
      }
    }
    vTaskDelay(1 / portTICK_PERIOD_MS); // small delay to yield
  }
}
