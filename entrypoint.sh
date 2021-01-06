#!/bin/bash

INPUT_COMMAND="${1}"

set -euo pipefail

# Download the model from azure blob storage
mkdir -p ../models
python ../scripts/app/fetch_model.py model.pth ../models/model.pth

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
