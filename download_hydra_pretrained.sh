#!/bin/bash

echo 'Downloading HYDRA pretrained models ...'
wget -O hydra_pretrained.zip "https://www.dropbox.com/scl/fi/9578idwzil5a0w3aeoyk9/hydra_pretrained.zip?rlkey=84jlcxxhs31tpvfi0igdk8zuk&dl=0"
echo 'Unzip HYDRA pretrained models ...'
unzip hydra_pretrained.zip -d ./