import sys

from delta.config import config
from delta.imagery.imagery_dataset import ImageryDataset
from delta.ml.train import train
from delta.ml.model_parser import config_model

def setup_parser(subparsers):
    sub = subparsers.add_parser('train', help='Train a task-specific classifier.')
    sub.add_argument('model', help='File to save the network to.')
    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, train=True)

def main(options):
    images = config.images()
    labels = config.labels()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    if not labels:
        print('No labels specified.', file=sys.stderr)
        return 1
    tc = config.training()

    ids = ImageryDataset(images, labels, config.chunk_size(), config.output_size(), tc.chunk_stride)

    try:
        model, _ = train(config_model(ids.num_bands()), ids, tc)

        model.save(options.model)
    except KeyboardInterrupt:
        print()
        print('Training cancelled.')

    return 0
