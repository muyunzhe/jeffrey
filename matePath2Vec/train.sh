#! /bin/bash


cd ${MODEL_PATH} && ./metapath2vec -train ${DATA_PATH}/att_author_metapath_wl5_${EVENT_DAY} \
               -output ${DATA_PATH}/att_author_metapath_5_100dim_${EVENT_DAY}_vec \
               -pp 1 \
               -size 100 \
               -window 5
