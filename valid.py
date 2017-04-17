# @Author: Aryan Singh
# @Date:   2017-04-18 02:49:07
# @Last Modified by:   Aryan Singh
# @Last Modified time: 2017-04-18 02:50:10

import os
from shutil import copyfile

# make changes in the path
#program copies out ~33% of dataset as validation set
# modify the code to cut and paste the files
# Copies direct images. You will have to normalize again
os.makedirs("/home/slifer/Py/~/Py/Stylus-master/valid_data")
cpy_pat= r'/home/slifer/Py/~/Py/Stylus-master/valid_data'

directory1 = r'/home/slifer/Py/Stylus-master/dataset/Sample0'



for j in range(1,62):
    tem=j
    sec= tem%10
    tem/=10
    fir= tem%10;
    directory = directory1 + str(fir) + str(sec) +'/'
    cnt=1;
    #print directory
    for i in os.listdir(directory):
        if cnt%3==0:
            pat = directory+i
            print cpy_pat
            copyfile(pat,cpy_pat+'/'+i)
        cnt=cnt+1
