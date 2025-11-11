#include "get_frames.h"
#include <Arduino.h>

extern volatile bool frame_ready;

bool get_next_frame() {
  if (frame_ready) {
    frame_ready = false;
    return true;
  }
  return false;
}

uint8_t frame_buffer[28 * 28];
volatile bool frame_ready = false;

void serial_thread_task(void* param) {
  while (true) {
    if (Serial.available() >= 28 * 28) {
      for (int i = 0; i < 28 * 28; i++) {
        frame_buffer[i] = Serial.read();
      }
      frame_ready = true;
      Serial.println("Frame received");
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}
