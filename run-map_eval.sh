python map_eval.py \
        --mode            standard \
        --buffer          example/buffer.pkl \
        --annotation_file example/GTBOX-TT100K-512.txt \
        --detection_file  example/darknet-TinyX5_45000-nms02-TT100K.txt \
        --dict_file       example/FT-sign.dict \
        --confidence      0.5 \
        --iou             0.5 \
        --missbox         False \
        --scaletest       True
