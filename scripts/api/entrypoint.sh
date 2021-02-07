#!/bin/bash

INPUT_COMMAND="${1}"

set -euo pipefail

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
