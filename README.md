This repository contains two projects, a PlatformIO project (ESP32 flash) to run an inference loop for a TinyCNN classifier trained on MNIST and a python backend to train models


PlatformIO:
-load tensorflow-lite-micro library, inference loop, and capture loop onto ESP32
-main.cpp runs the main event loops for CNN classifier and frame reciever


Python:
-Train PyTorch models using lightning, quantize and prune them into lean C int8[]
1. Python/ML contains lightning modules that implmeent a TinyCNN on MNIST
2. Python/embedded contains code to quantize the model, ONNX and embed it as a C array of int8

There are two seperate python enviornments due to package incompatabilities.
you can create two virtual enviornemtns from the two requirements.txts in the python folder.

Usage
-python -m ML.cnn_module will train a CNN
-python -m ML.train will run a Ray tune sweep and find the best CNN
-python -m Embedded.embed_model -path_to_pytorch_model will save a pytorch model as an embeddable C array of int8
