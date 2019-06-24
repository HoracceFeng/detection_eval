#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:32:38 2018

@author: horacce
"""

import sys
import numpy as np
import os
# import cv2
import time


def get_cord(xml_adr):
    import xml.etree.cElementTree as ET
    tree = ET.parse(xml_adr)
    root = tree.getroot()
    ret = []
    for child in root.findall("object"):
        cate = child.find("name").text
        box = child.find("bndbox")
        xmin = box.find("xmin").text
        ymin = box.find("ymin").text
        xmax = box.find("xmax").text
        ymax = box.find("ymax").text
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        ret.append([cate, box])
    return ret



xml_dir = sys.argv[1]
xml_list = os.listdir(xml_dir)

for xml_adr in xml_list:
    # img_adr = xml_adr.replace('Annotations','JPEGImages').replace('xml','jpg')
    # im = cv2.imread(im_)
    ret = get_cord(os.path.join(xml_dir, xml_adr))
    for box in ret:
        img_name = xml_adr.replace('xml','jpg')
        print '{}\t{}\t{}\t{}\t{}'.format(img_name,'label:',box[0], 1.0, box[1])
