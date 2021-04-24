#!/bin/bash

if [ ! -z $1 ]; then   
    python generate_dataset.py --notebook
    python filter_dataset.py --notebook
else
    python generate_dataset.py
    python filter_dataset.py
fi