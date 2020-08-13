#! /bin/bash

BASETIME=`date -d '1 day ago' +%Y-%m-%d" "%H:%M:%S`;

if [ $# == "2" ]; then
    BASETIME=$1" "$2;
fi

EVENT_DAY=`date -d "$BASETIME" +%Y%m%d`;



function preprocess(){
    cat ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}_vec.txt | \
        ${PYTHON_BIN} get_vec.py > ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.vec
}

function build_tree(){
    ${PYTHON_BIN} ${BIN_PATH}/annoyTree_build.py \
        ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.vec \
        ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.ann \
        ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.pkl
}

function search_tree(){
    ${PYTHON_BIN} ${BIN_PATH}/annoyTree_search.py \
        ${DATA_PATH}/lexer_month_seg_out_${EVENT_DAY} \
        150 \
        0.4 \
        ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.ann \
        ${DATA_PATH}/title_pub_ngram_metapath_5_100dim_${EVENT_DAY}.pkl \
    > ${DATA_PATH}/month_pub_seg_simi_${EVENT_DAY}
}

preprocess
build_tree
search_tree


