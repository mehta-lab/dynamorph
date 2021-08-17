#!/bin/bash

# runs training using horovodrun from fry


horovodrun \
-np 4 \
-H localhost:4 \
--verbose \
python ./train_seg_horovod.py
