#!/bin/bash

echo 'Downloading CHITA pretrained models ...'
wget -O chita_pruned.zip "https://www.dropbox.com/scl/fi/t2tkksijz1igis1e629p9/chita_pruned.zip?rlkey=05tn71jahqnswukf2adm3hepn&dl=0"
  echo 'Unzip CHITA pruned models ...'
unzip chita_pruned.zip -d ./