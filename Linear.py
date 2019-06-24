import numpy as np
import argparse
import os, sys, time, ast
import cPickle as cpkl
import map_eval as me


def output_info(recalls, precisions, average_precisions, num_lib, dictionary):
    tag = 0.
    sumup = 0.

    total_annos = 0.
    total_dects = 0.
    total_hits  = 0.

    results = {}
    results['total'] = None

    for label_name in dictionary:

        if len(recalls[label_name]) !=0 :
            rcl = recalls[label_name][-1]
            if isinstance(rcl, float):
                rcl = round(rcl, 4)
            else:
                rcl = 0.
        else:
            rcl = 0.

        if len(precisions[label_name]) !=0 :
            prc = precisions[label_name][-1]
            if isinstance(prc, float):
                prc = round(prc, 4)
            else:
                prc = 0.
        else:
            prc = 0.

        apr = round(average_precisions[label_name], 4)
        num_anno, num_dect, num_hit = num_lib[label_name]
        num_anno = int(num_anno); 
        num_dect = int(num_dect); 
        num_hit = int(num_hit); 

        total_annos += num_anno
        total_dects += num_dect
        total_hits  += num_hit             
   
        if prc != 'X' :
            tag += 1
            sumup += apr

        results[label_name] = [apr, prc, rcl]

    results['total'] = [round(sumup/len(average_precisions),3), round(total_hits/total_dects, 3), round(total_hits/total_annos, 3)]

    return results
      



if __name__ == '__main__':

    ### Parameters
    modes = ('standard', 'caliber')
    confidence_set = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]

    ### Arguments
    parser = argparse.ArgumentParser(description='P/R Line Module -- For precision/recall calculation in different confidence level')
    parser.add_argument('--mode',     choices=modes, default='caliber', type=str,
                        help="Choose one of the detection mode: "
                             " { %s }: generate P/R Line File for each category in the dictionary"
                             " { %s }: generate Total P/R Line File only" % modes)
    parser.add_argument('--buffer',          default=None,  type=str,    help='buffer pkl file contains the matrix created by annot_file & dects_file, \
                                                                               create by the name provided if not have')
    parser.add_argument('--annotation_file', default=None,  type=str,    help='ground truth box text file (annot_file), skip if have buffer')
    parser.add_argument('--detection_file',  default=None,  type=str,    help='detection box output text file (dects_file), skip if have buffer')
    parser.add_argument('--dict_file',       default=None,  type=str,    help='classes dictionary, detect annot and dects automatically if not provide')
    parser.add_argument('--imageset_file',   default=None,  type=str,    help='testset list file, if not provide, will use all img provided by annot_file')
    parser.add_argument('--iou',             default=0.5,   type=float,  help='IOU Threshold')
    parser.add_argument('--output_dir',      default=None,  type=str,    help='output_dir')
    parser.add_argument('--output_name',     default=None,  type=str,    help='output_name')
    args = parser.parse_args()


    ### buffer load/save and Check Arguments States
    ## buffer: contain dictionary, all_annots, annots_tag, all_dects, dects_tag, image_list
    if os.path.exists(args.buffer):
        _buffer         = cpkl.load(open(args.buffer, 'rb'))
        dictionary      = _buffer['dictionary']
        all_annotations = _buffer['all_annotations']
        annotations_tag = _buffer['annotations_tag']
        all_detections  = _buffer['all_detections']
        detections_tag  = _buffer['detections_tag']
        image_list      = _buffer['image_list']
        print "Pickle Exists, Loadin Success ......", args.buffer
    else:
        ## dictionary
        dictionary = []
        if args.dict_file is not None:
            _dict = open(args.dict_file, 'r').readlines()
            for cate in _dict:
                dictionary.append(cate.strip())
            print "Dictionary loading in ......", args.dict_file 
        else:
            print "Dictionary will be created by annots_file and dects_file"
  
        ## imageset_file
        image_list = []
        if args.imageset_file is not None and os.path.exists(args.imageset_file):
            _img_list = open(args.imageset_file, 'r').readlines()
            for _id in _img_list:
                image_list.append(_id.strip())

        ## annotation_file --> create ListObject `annots`
        if args.annotation_file is None:
            raise '[ Error ] please input parameter `annotation_file` '
        else:
            annf = open(args.annotation_file, 'r').readlines()
            annots = []
            for line in annf:
                imgid, _, cate, conf, bbox = line.strip().split('\t')
                imgid=imgid.strip()
                cate = cate.strip()
                conf = conf.strip()
                bbox = eval(bbox.strip())
                ## if no imageset_file, create it. 
                ## if have imageset_file but this img not in, this image will not be used 
                if args.imageset_file is None and imgid not in image_list:
                    image_list.append(imgid)
                ## if no dictionary, create it. 
                ## if have dictionary but the cateof this img is not, this image will not be used                     
                if args.dict_file is None and cate not in dictionary:
                    dictionary.append(cate)
                ## create list for later use
                annots.append([imgid, cate, conf, bbox])


        ## detection_file --> create ListObject `dects`
        if args.detection_file is None:
            raise '[ Error ] please input parameter `detection_file` '
        else:
            decf = open(args.detection_file, 'r').readlines()
            dects = []
            for line in decf:
                imgid, _, cate, conf, bbox = line.strip().split('\t')
                imgid=imgid.strip()
                cate = cate.strip()
                conf = conf.strip()
                bbox = eval(bbox.strip())
                ## if img not in image_list, this image will not be used
                if imgid not in image_list:
                    continue
                ## if dict_file not be appointed, then cate will be counted in dictionary
                ## else not process (may have bug), but we just use the dictionary of the model so that's OK
                if args.dict_file is None and cate not in dictionary:
                    dictionary.append(cate)
                dects.append([imgid, cate, conf, bbox])

        ## `dictionary, all_annots, annots_tag, all_dects, dects_tag, image_list` are all set 
        all_detections, detections_tag = me._get_detections(dects, dictionary, image_list, scales)
        all_annotations, annotations_tag = me._get_annotations(annots, dictionary, image_list, scales)

        ## buffer create ##
        buffer_writer                    = {}
        buffer_writer['dictionary']      = dictionary
        buffer_writer['all_annotations'] = all_annotations
        buffer_writer['annotations_tag'] = annotations_tag
        buffer_writer['all_detections']  = all_detections
        buffer_writer['detections_tag']  = detections_tag
        buffer_writer['image_list']      = image_list
        cpkl.dump(buffer_writer, open(args.buffer, 'wb'))
        print "Pickle Not be detected, auto-create Success ......", args.buffer

    print \
    'P/R LineFile Script for detecion algorithm: \n\
     to change the parameter, try `python Linear.py --help` \n\
        Parameters: \n \
        buffer:            {} \n \
        annotation_file:   {} \n \
        detection_file:    {} \n \
        dict_file:         {} \n \
        iou:               {} \n \
        output_dir:        {} \n \
        output_name:       {} \n '.format(args.mode, args.buffer, args.annotation_file, args.detection_file, args.dict_file, args.iou, args.output_dir, args.output_name)

    print "ImageSet:  \t", len(image_list)
    print "Dictionary:\t", dictionary
    print "\n"


    outfile = os.path.join(args.output_dir, 'LINEAR_'+args.output_name.split('.')[0]+'_DectBoxes.lines')
    if os.path.exists(outfile):
        os.remove(outfile)
    dectbox_recorder = open(outfile, 'w')
    print " --> Linear File Created: [DectBoxes] ", outfile

    if args.mode == 'standard':
        files = dictionary
        for i, label_name in enumerate(dictionary):
            outfile = os.path.join(args.output_dir, 'LINEAR_'+args.output_name.split('.')[0]+'_'+label_name+'.lines')
            if os.path.exists(outfile):
                os.remove(outfile)
            globals()[files[i]] = open(outfile, 'w') 
            print " --> Linear File Created: ", outfile
        

    for confidence in confidence_set:

        recalls, precisions, average_precisions, num_lib, scale_result = me.evaluate(
            all_annotations=all_annotations, annotations_tag=annotations_tag,
            all_detections=all_detections, detections_tag=detections_tag,
            dictionary=dictionary, image_list=image_list, confidence=confidence,
            iou_threshold=args.iou, missbox=False, scaletest=False)
        results = output_info(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                            num_lib=num_lib, dictionary=dictionary)
   
        total_apr, total_prc, total_rcl = results['total']
        dectbox_recorder.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Conf:', float(confidence), 'AP:', float(total_apr), 'Prec:', float(total_prc), 'Recall:', float(total_rcl)))

        if args.mode == 'standard':
            for i, label_name in enumerate(dictionary):
                apr, prc, rcl = results[label_name]
                globals()[files[i]].write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Conf:', float(confidence), 'AP:', float(apr), 'Prec:', float(prc), 'Recall:', float(rcl)))
            


