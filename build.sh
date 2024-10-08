
###
 # @Author: Clark
 # @Email: haixuanwoTxh@gmail.com
 # @Date: 2024-09-20 16:01:22
 # @LastEditors: Clark
 # @LastEditTime: 2024-10-08 16:49:33
 # @Description: file content
###

#!/bin/bash

if [ ! -d "3rd_party" ]; then
    7z x 3rd_party.7z
fi

if [ ! -d "data" ]; then
    7z x data.7z
fi

mkdir build
cd build
cmake .. && make -j16
cd ..
./build/jh_tof_camera
