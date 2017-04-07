# @Author: amaneureka
# @Date:   2017-04-08 03:00:03
# @Last Modified by:   amaneureka
# @Last Modified time: 2017-04-08 03:18:01

#!/usr/bin/env bash

DATASET=http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz
FOLDER=./dataset
TARBALL=EnglishHnd.tgz
SUBDIR=English/Hnd/Img

if [ ! -d $FOLDER ]; then
	mkdir $FOLDER
	if [ ! -f $TARBALL ]; then
		wget -O $TARBALL $DATASET || bail
	fi
	tar --strip-components=3 -C $FOLDER -xvf $TARBALL $SUBDIR
fi
