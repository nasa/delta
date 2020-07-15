# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lists all avaiable commands.
"""
from delta.config import config

#pylint:disable=import-outside-toplevel

# we put this here because tensorflow takes so long to load, we don't do it unless we have to
def main_classify(options):
    from . import classify
    classify.main(options)

def main_train(options):
    from . import train
    train.main(options)

def main_mlflow_ui(options):
    from .import mlflow_ui
    mlflow_ui.main(options)

def setup_classify(subparsers):
    sub = subparsers.add_parser('classify', help='Classify images given a model.')
    config.setup_arg_parser(sub, ['general', 'io', 'dataset'])

    sub.add_argument('--prob', dest='prob', action='store_true', help='Save image of class probabilities.')
    sub.add_argument('--autoencoder', dest='autoencoder', action='store_true', help='Classify with the autoencoder.')
    sub.add_argument('--no-colormap', dest='noColormap', action='store_true',
                     help='Save raw classification values instead of colormapped values.')
    sub.add_argument('model', help='File to save the network to.')

    sub.set_defaults(function=main_classify)

def setup_train(subparsers):
    sub = subparsers.add_parser('train', help='Train a task-specific classifier.')
    config.setup_arg_parser(sub)
    sub.add_argument('--autoencoder', action='store_true',
                     help='Train autoencoder (ignores labels).')
    sub.add_argument('--resume', help='Use the model as a starting point for the training.')
    sub.add_argument('model', nargs='?', default=None, help='File to save the network to.')
    sub.set_defaults(function=main_train)

def setup_mlflow_ui(subparsers):
    sub = subparsers.add_parser('mlflow_ui', help='Launch mlflow user interface to visualize run history.')
    config.setup_arg_parser(sub, ['mlflow'])

    sub.set_defaults(function=main_mlflow_ui)


SETUP_COMMANDS = [setup_train, setup_classify, setup_mlflow_ui]
