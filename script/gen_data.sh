#!/bin/bash

# Run a series of tests to generate necessary data for analysis

# Clear directories
rm -rf output

# TODO: ADD IMAGE FILENAMES
sh script/run.sh rgb1
sh script/run.sh rgb2
sh script/run.sh rgb3
sh script/run.sh rgb3
sh script/run.sh grayscale1
sh script/run.sh grayscale2
sh script/run.sh grayscale3 
sh script/run.sh grayscale4
