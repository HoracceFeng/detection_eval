python Linear.py \
        --mode            standard \
        --buffer          example/buffer.pkl \
        --annotation_file example/GTBOX-TT100K-512.txt \
        --detection_file  example/darknet-TinyX5_45000-nms02-TT100K.txt \
        --dict_file       example/FT-sign.dict \
    	--iou             0.5 \
    	--output_dir      example/LINEAR \
    	--output_name     darknet-tiny-TT100K
