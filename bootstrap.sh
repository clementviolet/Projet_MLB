#! /bin/bash

sudo apt-get update

# Install R and R packages
echo Install R and R packages

sudo apt-get install -y r-base
sudo Rscript -e "install.packages(c(\"missMDA\", \"tidyverse\"), repos = \"http://cran.us.r-project.org\")"
sudo Rscript ./script.R

# Install Python packages
echo Install Python packages

pip install --upgrade pip
pip install numpy
pip install setuptools
pip install --upgrade numpy setuptools # Causing issues to install tensorflow
pip install -r ./packages_py.txt