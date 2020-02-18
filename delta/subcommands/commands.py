from . import classify, train, mlflow_ui

SETUP_COMMANDS = [train.setup_parser,
                  classify.setup_parser,
                  mlflow_ui.setup_parser,
                 ]
