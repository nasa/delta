# Network Models

## autoencoder_conv.yaml 

| Filename | Model Description | Chunk Size Constraints |  
|----------|-------------------|------------------------|
| autoencoder_conv.yaml | Simple convolutional autoencoder. | Chunk sizes must be a multiple of 4|
| conv_autoencoder_128_chunk.yaml | An autoencoder with skip-links, based on the model presented on [this github](https://github.com/arahusky/Tensorflow-Segmentation) | Chunk sizes must be a multiple of 8, Can handle up to 128-pixel chunks |
| segnet-short.yaml | A simple one-layer convolutional autoencoder based on segnet. | Chunks must be a multiple of 2 |
| segnet-medium.yaml | Convolutional autoencoder based on segnet. | chunks must be a multiple of 4 |
| segnet.yaml | A full implementation of segnet. | Chunks must be a multiple of 32.  WARNING: Causes the model_parser to have a recursion error. |
