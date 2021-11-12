# Network Models

DELTA includes some example neural network architectures you can use in your project. They're included as .yaml files you can reference in the config files you pass to DELTA.

Excerpt from an example config file:

```yaml
train:
  network:
    # Create your own custom architecture
    yaml_file: path/to/your/custom/architecture/network_architecture.yaml
    # Or use an existing architecture included with DELTA
    yaml_file: path_to_delta_installation/delta/config/networks/segnet.yaml
```

The below table is an example of some of the architectures included with DELTA. You can find all of them in [delta/config/networks](./).

| Filename | Model Description | Chunk Size Constraints |  
|----------|-------------------|------------------------|
| autoencoder_conv.yaml | Simple convolutional autoencoder. | Chunk sizes must be a multiple of 4|
| conv_autoencoder_128_chunk.yaml | An autoencoder with skip-links, based on the model presented on [this github](https://github.com/arahusky/Tensorflow-Segmentation) | Chunk sizes must be a multiple of 8, Can handle up to 128-pixel chunks |
| segnet-short.yaml | A simple one-layer convolutional autoencoder based on segnet. | Chunks must be a multiple of 2 |
| segnet-medium.yaml | Convolutional autoencoder based on segnet. | chunks must be a multiple of 4 |
| segnet.yaml | A full implementation of segnet. | Chunks must be a multiple of 32.  WARNING: Causes the model_parser to have a recursion error. |

Some of these models are based on the SegNet auto-encoder based classifier, which is documented in this [[PDF](https://arxiv.org/pdf/1511.00561.pdf)]
