#!/bin/bash

INPUT_COMMAND="${1}"

set -euo pipefail

# Download the model from azure blob storage
mkdir -p models
python scripts/api/fetch_model.py model.pth models/model.pth

if [[ ${INPUT_COMMAND} == "bash" ]]; then
    shift
    exec "/bin/bash" "${@}"
elif [[ ${INPUT_COMMAND} == "python" ]]; then
    shift
    exec "python" "${@}"
elif [[ ${INPUT_COMMAND} == "start-uvicorn" ]]; then
    shift
    exec "uvicorn" "api.main:app" "--host" "0.0.0.0" "--port" "5000" "${@}"
# Debugging
else
    exec "${@}"
fi
