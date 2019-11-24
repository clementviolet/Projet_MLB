#! /bin/bash

sudo apt-get update

# Install R and R packages

sudo apt-get install -y r-base
sudo Rscript -e "install.packages(c("missMDA", "tidyverse"))"

# Install Python packages
pip install --upgrade pip
pip install -r ./packages_py.txt