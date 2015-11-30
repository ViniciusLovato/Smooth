#!/bin/bash

# Run a series of tests to generate necessary data for analysis

# Clear directories
rm -rf output

# TODO: ADD IMAGE FILENAMES
sh script/run.sh rgb1
sh script/run.sh rgb2
sh script/run.sh rgb3
sh script/run.sh rgb3
sh script/run.sh gray1
sh script/run.sh gray2
sh script/run.sh gray3
sh script/run.sh gray4
