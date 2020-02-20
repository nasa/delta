'''
.. include:: ../README.md
'''

import re
import sys
import tensorflow

if sys.version_info < (3, 0, 0):
    raise ImportError('DELTA code requires Python version >= 3.0.  Installed is %s' % (sys.version_info,))
