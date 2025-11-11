#include "get_frames.h"
#include <Arduino.h>

uint8_t frame_buffer[FRAME_SIZE];      
volatile bool frame_ready = false;     


// Returns true if a full frame is ready
bool get_next_frame() {
    if (frame_ready) {
        frame_ready = false;
        return true;
    }
    return false;
}

// Read bytes from serial into frame_buffer
void serial_thread_task(void* param) {
  static uint16_t index = 0; // current write position in frame_buffer

  while (true) {
    while (Serial.available() > 0) {
      uint8_t byte = Serial.read();
      frame_buffer[index++] = byte;

      // Check if a full frame is received
      if (index >= FRAME_SIZE) {
        frame_ready = true;  // mark frame ready
        index = 0;           // reset for next frame
        Serial.println("Frame received");
      }
    }
    vTaskDelay(1 / portTICK_PERIOD_MS); // small delay to yield
  }
}
