#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pip install pylint

cp $DIR/pre-commit $DIR/../../.git/hooks/pre-commit
echo "Linter installed as pre-commit hook."

