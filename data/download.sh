#!/bin/bash

TARGET_DIR=$HOME/Documents/datasets

# VANH
URL=https://redwood.berkeley.edu/cadieu/data/vid075-chunks.tar.gz
wget $URL $TARGET_DIR

# DAVIS
URL=https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-Unsupervised-trainval-480p.zip
wget $URL $TARGET_DIR
unzip $TARGET_DIR/DAVIS-2017-Unsupervised-trainval-480p.zip
rm $TARGET_DIR/DAVIS-2017-Unsupervised-trainval-480p.zip
# remove duplicate frames
DIR=$TARGET_DIR/DAVIS/JPEGImages/480p
mkdir $DIR/pigs-duplicate
for t in 25 50 75;
do
mv $DIR/pigs/000${t}.jpg $DIR/pigs-duplicate/000${t}.jpg
done
