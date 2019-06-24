import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os,sys,re
import numpy as np
import argparse


def file_reader(filepath, confidence_set):
    confs = []; aps = []; precs = []; recls = []
    files = open(filepath).readlines()
    for line in files:
        _, conf, _, ap, _, prec, _, recl = line.strip().split('\t')
        confs.append(float(conf))
        aps.append(float(ap))
        precs.append(float(prec))
        recls.append(float(recl))
    assert confidence_set == confs
    return confs, aps, precs, recls


def file_filter(filelist, keywords=[]):
    '''
    keywords:   ListObject to filter files which contains the keywords
    filelist:   ListObject, contains all path of target file
    '''
    if len(keywords) == 0:
        objects = filelist
        return objects
    else:
        objects = []

    for filename in filelist:
        outflag = True
        for word in keywords:
            if filename.find(word) == -1:
                outflag = False

        if outflag:
            objects.append(filename)

    return objects



if __name__ == '__main__':
    
    ## Arguments ##
    parser = argparse.ArgumentParser(description='Draw Line after generating .line file')
    parser.add_argument('-p', '--path',                type=str,             help='directory of .line file')
    parser.add_argument('-k', '--keywords', nargs='+', type=str, default=[], help='choose which lines you wanna draw, input keywords')
    args = parser.parse_args()         

    ## Parameters ##
    # linearpath     = 'result-for-dect_algo-eval-project/LINEAR/'
    # keywords       = ['DectBoxes', 'GIStest']   ## [dataset, model, epoch, step, etc], example ['pl']
    linearpath     = args.path
    keywords       = args.keywords
    linearlist     = os.listdir(linearpath)
    confidence_set = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]


    ## Linear Filter ##
    objects = file_filter(filelist=linearlist, keywords=keywords)

    print ("Linear File Filter:")
    print ("Linear File Detected:", len(objects))
    print ("==> ", objects)


    ### Line Drawer ###
    ### set random set for color
    NCURVES = len(objects)
    np.random.seed(101)
    values = range(NCURVES)
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)


    ## Drawing ##
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idx, obj in enumerate(sorted(objects)):
        objfile = os.path.join(linearpath, obj)
        confs, aps, precs, recls = file_reader(objfile, confidence_set)
        colorval = scalarMap.to_rgba(values[idx])
        ## 'label' ==> legend editor 
        label = ''
        for string in obj.strip().split('_')[1:-1]:
            label += string
        ax.plot(precs, recls, color=colorval, linewidth=2, marker='o', label=label)


    ### Lines ###
    ax.legend()
    plt.title('Prec-Recall Curves '+str(keywords))
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.axhline(y=0.98, c='black', linestyle='--', linewidth=1)   ## SOTA recall
    ax.axvline(x=0.80, c='black', linestyle='--', linewidth=1)   ## SOTA precision

    ax.axhline(y=0.955, c='black', linestyle='--', linewidth=1)  ## Robust recall
    ax.axvline(x=0.90, c='black', linestyle='--', linewidth=1)   ## Robust precision

    ax.axhline(y=0.99, c='black', linestyle='--', linewidth=1)   ## Analyz recall
    ax.axvline(x=0.91, c='black', linestyle='--', linewidth=1)   ## Analyz precision
    ax.axvline(x=0.92, c='black', linestyle='--', linewidth=1)   ## Analyz precision
    ax.axvline(x=0.93, c='black', linestyle='--', linewidth=1)   ## Analyz precision
    ax.axvline(x=0.94, c='black', linestyle='--', linewidth=1)   ## Analyz precision

    ax.set_xlim(0.6, 1)    ## x-precision
    ax.set_ylim(0.85, 1)    ## y-recall
    plt.show()




