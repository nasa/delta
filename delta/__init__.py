'''
The DELTA project is designed to
'''

import re
import sys
import tensorflow

if sys.version_info < (3, 0, 0):
    print('\nERROR: DELTA code requires Python version >= 3.0.')
    sys.exit(1)

if re.search('^1.12', tensorflow.__version__) is None:
    print('\nERROR: DELTA code requires Tensorflow version 1.12.0')
    sys.exit(1)
