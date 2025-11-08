"""Quick test: instantiate LitCNN and run a forward pass on dummy data.

Run this file with the project's Python environment to sanity-check the module.
"""

import torch

from ML.cnn_module import LitCNN


def main():
    config = {
        'input_channels': 1,
        'num_classes': 10,
        'conv_layers': [
            {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': 2},
            {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'pool': 2},
        ],
        'fc_layers': [32],
        'activation': 'relu',
        'dropout': 0.0,
        'lr': 1e-3,
        'optimizer': 'adam',
    }

    model = LitCNN(config)
    model.eval()

    # batch of 4, channels=1, 28x28 (like MNIST)
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    print('logits.shape =', logits.shape)

    assert logits.shape == (4, config['num_classes'])
    print('Sanity check passed.')


if __name__ == '__main__':
    main()
