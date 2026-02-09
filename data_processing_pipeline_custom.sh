#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m preprocess.data_processing_pipeline_custom \
    --total_num_workers 16 \
    --per_gpu_num_workers 16 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir temp \
    --input_dir ./data/vfhq_motion/
