'''
The DELTA project is designed to
'''

import re
import sys
import tensorflow

if sys.version_info < (3, 0, 0):
    raise ImportError('DELTA code requires Python version >= 3.0.  Installed is %s' % (sys.version_info,))

if re.search('^2.0', tensorflow.__version__) is None:
    raise ImportError('DELTA code requires Tensorflow version 2.0 installed: %s' %(tensorflow.__version__, ))
