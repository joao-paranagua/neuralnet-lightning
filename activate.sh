#!/bin/bash
if [ ! -d "neuralnet-env" ]; then
    echo "Virtual environment not found. Run 'make venv' first."
else
    source neuralnet-env/bin/activate
    echo "Virtual environment activated."
fi
