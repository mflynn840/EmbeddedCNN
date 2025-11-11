#pragma once
#include <Arduino.h>

void serial_thread_task(void* param);
extern uint8_t frame_buffer[28 * 28];
extern volatile bool frame_ready;
