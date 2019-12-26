from . import autoencode, classify, train

SETUP_COMMANDS = [autoencode.setup_parser,
                  train.setup_parser,
                  classify.setup_parser,
                 ]
