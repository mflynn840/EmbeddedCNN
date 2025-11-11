#pragma once
#include <Arduino.h>
#define FRAME_SIZE 28*28


void serial_thread_task(void* param);
bool get_next_frame();
extern uint8_t frame_buffer[FRAME_SIZE];
extern volatile bool frame_ready;
