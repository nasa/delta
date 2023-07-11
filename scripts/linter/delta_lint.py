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

from pylint.interfaces import IRawChecker
from pylint.checkers import BaseChecker

COPYRIGHT_HEADER = \
"""
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
""".strip().split('\n')

class CopyrightChecker(BaseChecker):
    __implements__ = IRawChecker

    name = 'copyright'
    msgs = {'W9901': ('Add the copyright header.',
                      'file-no-copyright',
                      ('Copyright message missing.')),
            }
    options = ()

    def process_module(self, node):
        offset = 0
        with node.stream() as stream:
            for (lineno, line) in enumerate(stream):
                line = line.decode('UTF-8').strip()
                if lineno - offset >= len(COPYRIGHT_HEADER):
                    break
                if line != COPYRIGHT_HEADER[lineno - offset].strip():
                    if lineno < 5:
                        offset += 1
                        continue
                    self.add_message('file-no-copyright',
                                     line=lineno)
                    break

def register(linter):
    linter.register_checker(CopyrightChecker(linter))
