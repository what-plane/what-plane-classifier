#!/bin/bash

INPUT_COMMAND="${1}"

set -euo pipefail

# Download the model from azure blob storage
mkdir models
# TODO replace this with python API?
az storage blob download -c models -f models/model.pth -n model.pth

if [[ ${INPUT_COMMAND} == "bash" ]]; then
    shift
    exec "/bin/bash" "${@}"
elif [[ ${INPUT_COMMAND} == "python" ]]; then
    shift
    exec "python" "${@}"
# Debugging
else
    exec "${@}"
fi
