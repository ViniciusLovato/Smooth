#!/bin/bash

# Run a series of tests to generate necessary data for analysis

# Clear directories
rm -rf output

# TODO: ADD IMAGE FILENAMES
sh ./run.sh rgb1
sh ./run.sh rgb2
sh ./run.sh rgb3
sh ./run.sh rgb3
sh ./run.sh grayscale1
sh ./run.sh grayscale2
sh ./run.sh grayscale3 
sh ./run.sh grayscale4
