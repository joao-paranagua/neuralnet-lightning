#!/bin/bash
if [ ! -d "cern" ]; then
    echo "Virtual environment not found. Run 'make venv' first."
else
    source cern/bin/activate
    echo "Virtual environment activated."
fi
