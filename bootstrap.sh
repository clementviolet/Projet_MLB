#! /bin/bash

sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/'
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo apt update

# Install R and R packages
echo Install R and R packages

sudo apt-get install -y r-base-core=3.5.2-1xenial0
sudo Rscript -e "install.packages(c(\"missMDA\", \"readr\", \"forcats\"), repos = \"http://cran.us.r-project.org\")"
sudo Rscript ./script.R

# Install Python packages
echo Install Python packages

pip install --upgrade pip
pip install numpy
pip install setuptools
pip install --upgrade numpy setuptools # Causing issues to install tensorflow
pip install -r ./packages_py.txt