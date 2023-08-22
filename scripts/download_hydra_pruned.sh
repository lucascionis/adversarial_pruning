#!/bin/bash

echo 'Downloading HYDRA pretrained models ...'
wget -O hydra_pruned.zip "https://www.dropbox.com/scl/fi/pdblm8rqhvdrrew0a9g6q/hydra_pruned.zip?rlkey=yg2ey8pgimopwrpgqst5q2qwz&dl=0"
echo 'Unzip HYDRA pruned models ...'
unzip hydra_pruned.zip -d ./