#!/bin/bash

echo 'Downloading HARP pruned models ...'
wget -O harp_pruned.zip "https://www.dropbox.com/scl/fi/rjxcnjmew7lis8hnnag6m/harp_pruned.zip?rlkey=esz3wg6sym7pqcgbey2ocpxf7&dl=0"
echo 'Unzip HARP pruned models ...'
unzip harp_pruned.zip -d ./