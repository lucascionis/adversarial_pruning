#!/bin/bash

echo 'Downloading HARP pretrained models ...'
wget https://www.dropbox.com/s/itz7fxphtoqbbwf/trained_models.zip
echo 'Unzip HARP pretrained models ...'
mv trained_models.zip harp_pretrained.zip
unzip trained_models.zip -d ./