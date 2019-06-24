import numpy as np
import argparse
import os, sys, time, ast
import cPickle as cpkl



def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua    



def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def _get_annotations(annots, dictionary, img_list, scales=[16,32,48,64,96]):
    print "Start _get_annotations() ......"
    _all_annotation = []
    
    ### set-rules to process annots if necessary
    # for anno in annots:
    #     imgid, cate, conf, box = anno
    #     _all_annotation.append([imgid, cate, conf, box])
    _all_annotation = annots

    all_annotations = [[ [] for i in range(len(dictionary))] for j in range(len(img_list))]
    annotations_tag = [[ [] for i in range(len(dictionary))] for j in range(len(img_list))]

    for i in range(len(img_list)):
        img_name = img_list[i]
        for p in range(len(_all_annotation)):
            if _all_annotation[p][0] == img_name:
                ## data parser
                _id, _cate, _conf, _box = _all_annotation[p]
                if _cate not in dictionary:
                    continue
                xmin, ymin, xmax, ymax = _box
                xmin=int(xmin); ymin=int(ymin); xmax=int(xmax); ymax=int(ymax)
                boxscale = min(xmax-xmin, ymax-ymin)
                _tag = None
            else:
                continue

            ## scaleid to _tag for scaletest calculation
            for scaleid, scale in enumerate(scales):
                if scaleid == 0:
                    if boxscale > scale:
                        continue
                    else:
                        _tag = scaleid + 1
                elif scaleid == len(scales)-1:
                    if boxscale > scale:
                        _tag = scaleid + 2
                    else:
                        continue
                elif scales[scaleid-1] > boxscale or scale < boxscale:
                    continue
                else:
                    _tag = scaleid + 1

            if _tag == None:
                _tag = len(scales)

            ## output two matrixs, one with all detailed annnots, another for scaletest
            _label = dictionary.index(_cate)
            all_annotations[i][_label].append(_box)
            annotations_tag[i][_label].append(_tag)
            
    return all_annotations, annotations_tag




def _get_detections(dects, dictionary, img_list, scales=[16,32,48,64,96]):
    print "Start _get_detections() ......"
    _all_detection = []

    ## set-rules to process dects if necessary
    # for dect in dects:
    #     imgid, cate, score, bbox = dect
    #     _all_detection.append([imgid, cate, score, bbox])
    _all_detection = dects

    all_detections = [[ [] for i in range(len(dictionary))] for j in range(len(img_list))]
    detections_tag = [[ [] for i in range(len(dictionary))] for j in range(len(img_list))]

    for i in range(len(img_list)):
        img_name = img_list[i]
        for p in range(len(_all_detection)):
            if _all_detection[p][0] == img_name:
                ## dect parser
                _id, _cate, _score, _box = _all_detection[p]
                if _cate not in dictionary:
                    continue
                xmin,ymin,xmax,ymax = _box
                xmin=int(xmin); ymin=int(ymin); xmax=int(xmax); ymax=int(ymax)
                boxscale = min(xmax-xmin, ymax-ymin)
                _tag = None
            else:
                continue

            ## scaleid to _tag for scaletest calculation
            for scaleid, scale in enumerate(scales):
                if scaleid == 0:
                    if boxscale > scale:
                        continue
                    else:
                        _tag = scaleid + 1
                elif scaleid == len(scales)-1:
                    if boxscale > scale:
                        _tag = scaleid + 2
                    else:
                        continue
                elif scales[scaleid-1] > boxscale or scale < boxscale:
                    continue
                else:
                    _tag = scaleid + 1
            if _tag == None:
                _tag = len(scales)

            ## output two matrixs, one with all detailed dects, one for scaletest
            _label = dictionary.index(_cate)
            all_detections[i][_label].append([_box, _score])
            detections_tag[i][_label].append(_tag)
            
    return all_detections, detections_tag
    

def scale_filter(boxset, tagset, scale_id):
    newboxset = []
    if len(boxset) != len(tagset):
        raise 'Error: boxset cannot match tagset'
    for box_id, box in enumerate(boxset):
        tag = tagset[box_id]
        if tag != scale_id:
            continue
        else:
            newboxset.append(box)
    return newboxset


def confd_filter(boxset, tagset, conf_thresh):
    outboxs = []
    outtags = []
    if len(boxset) == 0:
        return boxset, tagset
    for idx, objects in enumerate(boxset):
        _box, _score = objects
        if float(_score) >= conf_thresh:
            outboxs.append(objects)
            outtags.append(tagset[idx])
    return outboxs, outtags


def box_relations_via_labels(dictionary, label, image_list, all_detections, all_annotations, annotations_tag, detections_tag, \
                 conf_thresh, iou_threshold=0.5, scale_id=False, missbox_recorder=None):
    '''
    missbox_recorder   : to record missbox images, `missbox_recorder` object must be an object with attribute `write`
    '''
    label_name = dictionary[label]
    false_positives = np.zeros((0,))
    true_positives  = np.zeros((0,))
    scores          = np.zeros((0,))
    num_annotations = 0.0
    num_detections  = 0.0
    num_hit         = 0.0

    for i in range(len(image_list)):

        if scale_id == False:
            ## origin, no filter
            _detections                  = all_detections[i][label]
            _annotations                 = all_annotations[i][label]
        else:
            ## scale-test filter
            _detections                  = scale_filter(all_detections[i][label], detections_tag[i][label], scale_id)
            _annotations                 = scale_filter(all_annotations[i][label], annotations_tag[i][label], scale_id)

        ## confidence filter
        _detections, _detections_tag     = confd_filter(_detections, detections_tag[i][label], conf_thresh) 

        detections                       = np.array(_detections)
        annotations                      = np.array(_annotations)
        num_annotations                 += annotations.shape[0]
        num_detections                  += detections.shape[0]
        detected_annotations             = []

        ### evaluation measurements calculator ###
        for d_id, d in enumerate(detections):
            scores = np.append(scores, float(d[1]))
            _d = []
            for numb in d[0]:
                ## xmin,ymin,xmax,ymax
                _d.append(float(numb))
            ## confidence
            _d.append(float(d[1]))
            d = _d

            if annotations.shape[0] == 0:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
                continue

            overlaps            = compute_overlap(np.expand_dims(np.array(d), axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap         = overlaps[0, assigned_annotation]

            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
                num_hit += 1
            else:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)
                if missbox_recorder is not None:
                    missbox_recorder.write('{}\t{}\t{}\n'.format(image_list[i], label_name, d))

    return false_positives, true_positives, scores, num_annotations, num_detections, num_hit


def measurements_calculator(false_positives, true_positives, scores, num_annotations):
    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        recall = 'X'
        precision = 'X'
        average_precision = 0
        return recall, precision, average_precision

    # sort by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    # print "precision calculate", true_positives, np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision  = _compute_ap(recall, precision)

    return recall, precision, average_precision



def evaluate(
    all_annotations, annotations_tag,
    all_detections, detections_tag,
    dictionary,
    image_list,
    confidence,
    iou_threshold=0.5,
    missbox=False,
    scaletest=False,    
    scales=[16,32,48,64,96]):
    """ Evaluate a given dataset using a given model.

    # Arguments
        image_list      : The list of testset images
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    '''
    take out these four variables
    '''
    recalls = {}; precisions = {}; average_precisions = {}; num_lib = {}

    if missbox:
    	missbox_file = open('MISSBOX.txt', 'w')
    else:
    	missbox_file = None

    for label in range(len(dictionary)):
        label_name = dictionary[label]
        false_positives, true_positives, scores, num_annotations, num_detections, num_hit = \
            box_relations_via_labels(dictionary=dictionary, label=label, image_list=image_list, 
                                    all_detections=all_detections, all_annotations=all_annotations, 
                                    annotations_tag=annotations_tag, detections_tag=detections_tag, 
                                    conf_thresh=confidence, iou_threshold=iou_threshold, scale_id=False, missbox_recorder=missbox_file)

        num_lib[label_name] = [num_annotations, num_detections, num_hit]   ## format [#anno, #dect, #hit] 
        recall, precision, average_precision = measurements_calculator(false_positives, true_positives, scores, num_annotations)

        recalls[label_name] = recall
        precisions[label_name] = precision
        average_precisions[label_name] = average_precision

    if missbox:
    	missbox_file.close()

    scale_result = {}
    if scaletest:
        min_tag = 1; max_tag = len(scales)+1

        for tagg in range(min_tag, max_tag+1):
            _num_lib = {}
            exec('recalls_'+str(tagg)+'={}')
            exec('precisions_'+str(tagg)+'={}')
            exec('average_precisions_'+str(tagg)+'={}')
            exec('num_lib_'+str(tagg)+'={}')
            variable = ['recalls_'+str(tagg), 'precisions_'+str(tagg), 'average_precisions_'+str(tagg), 'num_lib_'+str(tagg)]

            for label in range(len(dictionary)):
                label_name = dictionary[label]
                false_positives, true_positives, scores, num_annotations, num_detections, num_hit = \
                    box_relations_via_labels(dictionary=dictionary, label=label, image_list=image_list, 
                                            all_detections=all_detections, all_annotations=all_annotations, 
                                            annotations_tag=annotations_tag, detections_tag=detections_tag, 
                                            conf_thresh=confidence, iou_threshold=iou_threshold, scale_id=tagg, missbox_recorder=None)

                _num_lib = [num_annotations, num_detections, num_hit]   ## format [#anno, #dect, #hit] 
                recall, precision, average_precision = measurements_calculator(false_positives, true_positives, scores, num_annotations)

                if recall == 'X':
                    exec("eval(variable[0])['"+label_name+"']='"+recall+"'")
                    exec("eval(variable[1])['"+label_name+"']='"+precision+"'")
                else:
                    exec("eval(variable[0])['"+label_name+"']=recall")
                    exec("eval(variable[1])['"+label_name+"']=precision")
                exec("eval(variable[2])['"+label_name+"']=average_precision")
                exec("eval(variable[3])['"+label_name+"']=_num_lib")

            exec("scale_result["+str(tagg)+"]=[eval(variable[0]), eval(variable[1]), eval(variable[2]), eval(variable[3])]")

    print '\n'
    return recalls, precisions, average_precisions, num_lib, scale_result


def output_controller(recalls, precisions, average_precisions, num_lib, dictionary, cal_dect=False, cal_scale=False):
    '''
    by changing the parameter `dictionary`, you can choose to calculate the AP of specific set of classes
    '''
    tag = 0.
    sumup = 0.

    total_annos = 0.
    total_dects = 0.
    total_hits  = 0.

    for label_name in dictionary:

        if len(recalls[label_name]) !=0 :
            rcl = recalls[label_name][-1]
            if isinstance(rcl, float):
                rcl = round(rcl, 4)
        else:
            rcl = 0.

        if len(precisions[label_name]) !=0 :
            prc = precisions[label_name][-1]
            if isinstance(prc, float):
                prc = round(prc, 4)
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

        if not cal_dect:
            print '{0}\t{1}:{2:<5}\t{3}:{4:<5}\t{5}:{6:<5}\t\t{7}:{8:<5}\t{9}:{10:<5}\t{11}:{12:<5}'.format(label_name, '#Annot', str(num_anno), \
                '#Dects', str(num_dect), '#Hit', str(num_hit), '#AP', str(apr), '#Precision', str(prc), '#Recall', str(rcl))        
        if prc != 'X' :
            tag += 1
            sumup += apr

    ### Result Summary
    if not cal_dect:
        print '\n'
        print     'tag (have gt-boxes class):', tag
        print     'sum:                      ', sumup
        if tag > 0:
            print 'Sum #Anno:{}\tSum #Dects:{}\tSum #Hits:{}\t'.format(str(int(total_annos)), str(int(total_dects)), str(int(total_hits)))
            print 'Total Precision:{}\tTotal Recall:{}\t'.format(str(round(total_hits/total_dects, 3)), str(round(total_hits/total_annos, 3)))
            if not cal_scale:
                print '\n'
                print 'Whole      mAP [{} type]:\t{}'.format(str(len(dictionary)), str(round(sumup/len(average_precisions),3)))
                print 'Non-sparse mAP [{} type]:\t{}'.format(str(int(tag)), str(round(sumup/tag,3)))
        else:
            print 'No ground-truth box. Skip it!'
        print '==========================================================================================================='
    else:
        if tag > 0:
            print 'Sum #Anno:{}\tSum #Dects:{}\tSum #Hits:{}\t'.format(str(int(total_annos)), str(int(total_dects)), str(int(total_hits)))
            print 'Total Precision:{}\tTotal Recall:{}\t'.format(str(round(total_hits/total_dects, 3)), str(round(total_hits/total_annos, 3)))
            if not cal_scale:
                print '\n'
                print 'Whole      mAP [{} type]:\t{}'.format(str(len(dictionary)), str(round(sumup/len(average_precisions),3)))
                print 'Non-sparse mAP [{} type]:\t{}'.format(str(int(tag)), str(round(sumup/tag,3)))
        else:
            print 'No ground-truth box. Skip it!'
        print '==========================================================================================================='        







if __name__ == '__main__':

    ### Parameters
    scales=[16,32,48,64,96]
    modes = ('standard', 'caliber')

    ### Arguments
    parser = argparse.ArgumentParser(description='Evaluation Module -- For precision, recall and mAP calculation')
    parser.add_argument('--mode',     choices=modes, default='caliber', type=str,
                        help="Choose one of the detection mode: "
                             " { %s }: calculate `all class mAP` only"
                             " { %s }: calculate both `all class mAP` and `dect box AP`" % modes)
    parser.add_argument('--buffer',          default=None,  type=str,    help='buffer pkl file contains the matrix created by annot_file & dects_file, \
                                                                               create by the name provided if not have')
    parser.add_argument('--annotation_file', default=None,  type=str,    help='ground truth box text file (annot_file), skip if have buffer')
    parser.add_argument('--detection_file',  default=None,  type=str,    help='detection box output text file (dects_file), skip if have buffer')
    parser.add_argument('--dict_file',       default=None,  type=str,    help='classes dictionary, detect annot and dects automatically if not provide')
    parser.add_argument('--imageset_file',   default=None,  type=str,    help='testset list file, if not provide, will use all img provided by annot_file')
    parser.add_argument('--confidence',      default=0.1,   type=float,  help='Confidence Threshold')
    parser.add_argument('--iou',             default=0.5,   type=float,  help='IOU Threshold')
    parser.add_argument('--missbox',         default=False, type=ast.literal_eval, help='output MISS-GTBOX recorder file')
    parser.add_argument('--scaletest',       default=False, type=ast.literal_eval, help='boxscale test')
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
        all_detections, detections_tag = _get_detections(dects, dictionary, image_list, scales)
        all_annotations, annotations_tag = _get_annotations(annots, dictionary, image_list, scales)

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
    'Evaluation Script for detecion algorithm: \n\
     to change the parameter, try `python map_eval.py --help` \n\
        Parameters: \n \
        mode:              {} \n \
        buffer:            {} \n \
        annotation_file:   {} \n \
        detection_file:    {} \n \
        dict_file:         {} \n \
        confidence:        {} \n \
        iou:               {} \n \
        missbox:           {} \n \
        scaletest:         {} \n '.format(args.mode, args.buffer, args.annotation_file, args.detection_file, args.dict_file, args.confidence, args.iou, args.missbox, args.scaletest)

    print "ImageSet:  \t", len(image_list)
    print 'Scales:    \t', scales 
    print "Dictionary:\t", dictionary


    if args.mode == 'caliber':

        confidence_set = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
        if not args.scaletest:
            for confidence in confidence_set:
                print '==========================================================================================================='
                print 'Confidence:', confidence
                recalls, precisions, average_precisions, num_lib, scale_result = evaluate(
                    all_annotations=all_annotations, annotations_tag=annotations_tag,
                    all_detections=all_detections, detections_tag=detections_tag,
                    dictionary=dictionary, image_list=image_list, confidence=confidence,
                    iou_threshold=args.iou, missbox=args.missbox, scaletest=False, scales=scales)
                output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                    num_lib=num_lib, dictionary=dictionary, cal_dect=True, cal_scale=False)
        else:
            for confidence in confidence_set:
                print '==========================================================================================================='
                print 'Confidence:', confidence
                recalls, precisions, average_precisions, num_lib, scale_result = evaluate(
                    all_annotations=all_annotations, annotations_tag=annotations_tag,
                    all_detections=all_detections, detections_tag=detections_tag,
                    dictionary=dictionary, image_list=image_list, confidence=confidence,
                    iou_threshold=args.iou, missbox=args.missbox, scaletest=True, scales=scales)
                output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                    num_lib=num_lib, dictionary=dictionary, cal_dect=True, cal_scale=False)            

                for scl in scale_result.keys():
                    recalls, precisions, average_precisions, num_lib = scale_result[scl]
                    if scl-1 < len(scales):
                        print '{}\t{}-{}'.format('Scale-Test Result', 'Scale', scales[scl-1])
                    else:
                        print '{}\t{}-{}'.format('Scale-Test Result', 'Scale-above', scales[-1])         
                    output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                        num_lib=num_lib, dictionary=dictionary, cal_dect=True, cal_scale=True)


    elif args.mode == 'standard':

        if not args.scaletest:
            recalls, precisions, average_precisions, num_lib, scale_result = evaluate(
                all_annotations=all_annotations, annotations_tag=annotations_tag,
                all_detections=all_detections, detections_tag=detections_tag,
                dictionary=dictionary, image_list=image_list, confidence=args.confidence,
                iou_threshold=args.iou, missbox=args.missbox, scaletest=False, scales=scales)
            output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                num_lib=num_lib, dictionary=dictionary, cal_dect=False, cal_scale=False)
        else:
            recalls, precisions, average_precisions, num_lib, scale_result = evaluate(
                all_annotations=all_annotations, annotations_tag=annotations_tag,
                all_detections=all_detections, detections_tag=detections_tag,
                dictionary=dictionary, image_list=image_list, confidence=args.confidence,
                iou_threshold=args.iou, missbox=args.missbox, scaletest=True, scales=scales)

            output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                num_lib=num_lib, dictionary=dictionary, cal_dect=False, cal_scale=False)

            for scl in scale_result.keys():
                recalls, precisions, average_precisions, num_lib = scale_result[scl]
                if scl-1 < len(scales):
                    print '{}\t{}-{}'.format('Scale-Test Result', 'Scale', scales[scl-1])
                else:
                    print '{}\t{}-{}'.format('Scale-Test Result', 'Scale-above', scales[-1])         
                output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                                    num_lib=num_lib, dictionary=dictionary, cal_dect=False, cal_scale=True)









