"""
Classify input images given a model.
"""
import os

from delta.config import config

def setup_parser(subparsers):
    sub = subparsers.add_parser('mlflow_ui', description='Launch mlflow user interface to visualize run history.')

    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, general=False, images=False, labels=False, ml=False)

def main(_):
    os.system('mlflow ui --backend-store-uri %s' % (config.mlflow_uri()))
    return 0
