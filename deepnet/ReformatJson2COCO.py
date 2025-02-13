# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: transformer
#     language: python
#     name: python3
# ---

# %%
import json
import numpy as np
import os
import cv2
import csv

# %%
def ConvertJson2COCO(infile,outfile,newimdir=None,extrainfo=None,isma=False,imdir=None,skeledges=None,kpnames=None,catname='animal',
                     behaviors=None,conditions=None,label_category_file=None):
    with open(infile,'r') as f:
        indata = json.load(f)
    outdata = {}
    if extrainfo is None:
        extrainfo = {}
    outdata['info'] = copy.deepcopy(extrainfo)
    outdata['info']['movies'] = indata['movies']
    outdata['annotations'] = []
    outdata['images'] = []
    if extrainfo is not None:
        for k,v in extrainfo.items():
            outdata['info'][k] = v
    for i in range(len(indata['locdata'])):
        x = indata['locdata'][i]
        z = {}
        z['id'] = i
        if newimdir is not None:
            filestr = os.path.split(x['img'][0])[-1]
            z['file_name'] = os.path.join(newimdir,filestr)
        else:
            z['file_name'] = x['img'][0]
        if isma:
            assert imdir is not None, 'Must provide imdir for isma data'
            imfile = os.path.join(imdir,z['file_name'])
            assert os.path.exists(imfile), 'Image file does not exist: %s' % imfile
            img = cv2.imread(imfile)
            z['width'] = img.shape[1]
            z['height'] = img.shape[0]
        else:
            z['width'] = x['roi'][2] # not sure how to make this more general
            z['height'] = x['roi'][3]
        outdata['images'].append(z)
        y = {}
        y['id'] = i
        y['image_id'] = i # not sure how this will work when multi-animal
        y['mov'] = x['imov']
        y['frm'] = x['frm']
        y['tgt'] = x['itgt']
        y['bbox'] = [x['roi'][0]-1,x['roi'][1]-1,x['roi'][2],x['roi'][3]]
        y['keypoints'] = []
        nkpts = len(x['pabs'])//2
        for j in range(nkpts):
            y['keypoints'].append(x['pabs'][j]-1)
            y['keypoints'].append(x['pabs'][nkpts+j]-1)
            v = 2-int(x['occ'][j])
            if np.isinf(x['pabs'][j]) or np.isnan(x['pabs'][j]) or \
                np.isinf(x['pabs'][nkpts+j]) or np.isnan(x['pabs'][nkpts+j]):
                    v = 0
            y['keypoints'].append(v)
        y['num_keypoints'] = nkpts
        y['category_id'] = 0 # only one category
        outdata['annotations'].append(y)

    if label_category_file is not None:
        # read in csv file
        mft = [(x['mov'],x['frm'],x['tgt']) for x in outdata['annotations']]
        with open(label_category_file,'r') as f:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                try:
                    v = tuple([int(x) for x in [row['mov'],row['frm'],row['tgt']]])
                    i = mft.index(v)
                except ValueError:
                    raise 'Could not find annotation for %s %s %s' % (row['mov'],row['frm'],row['tgt'])
                    continue
                outdata['annotations'][i]['behavior'] = int(row['behaviors'])
                outdata['annotations'][i]['condition'] = int(row['conditions'])
        if 'info' not in outdata:
            outdata['info'] = {}
        outdata['info']['behaviors'] = behaviors
        outdata['info']['conditions'] = conditions

    cat = {'id': 0, 'name': catname}
    if skeledges is None:
        skeledges = [[i,i+1] for i in range(nkpts-1)]
    if kpnames is None:
        kpnames = [f'kp{i}' for i in range(nkpts)]
    cat['keypoints'] = kpnames
    cat['skeleton'] = skeledges
    outdata['categories'] = [cat,]

    with open(outfile, 'w') as f:
        json.dump(outdata, f)


# %%
# infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20220525/multitarget_bubble_training_20210523_allGT_AR_params20210920/loc.json'
# outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data/train_annotations.json'
# newimdir = 'train'
# ConvertJson2COCO(infile,outfile,newimdir)

# infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20220525/test/loc.json'
# outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data/test_annotations.json'
# newimdir = 'test'
# ConvertJson2COCO(infile,outfile,newimdir)

# %%
# # cp -r /groups/branson/home/robiea/.apt/tpb4d19d6d_4b41_4a76_91d2_e37c6deaa229/multitarget_bubble_training_20210523_allGT_AR_params20210920 /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_train
# # mkdir /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024
# # cp -r /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_train/im /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/train
extrainfo = {'description': 'Fly Bubble training data', 'version': 1.0, 'contributor': 'Alice A. Robie and Kristin Branson','date_created': '2024-10-24'}
kpnames = [
    'ant_head',
    'right_eye',
    'left_eye',
    'left_thorax',
    'right_thorax',
    'pos_notum',
    'pos_abdomen',
    'right_mid_fe',
    'right_mid_fetib',
    'left_mid_fe',
    'left_mid_fetib',
    'right_front_tar',
    'right_mid_tar',
    'right_back_tar',
    'left_back_tar',
    'left_mid_tar',
    'left_front_tar',
    'right_mid_wing',
    'right_outer_wing',
    'left_mid_wing',
    'left_outer_wing',
]
skeledges = [
    [9,13],
    [10,11],
    [11,16],
    [6,10],
    [8,9],
    [6,8],
    [4,17],
    [5,12],
    [6,15],
    [6,14],
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [20,21],
    [18,19],
    [7,18],
    [7,20],
]

infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_train/loc.json'
outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/train_annotations.json'
newimdir = 'train'
imdir = os.path.dirname(outfile)
assert os.path.exists(imdir)
extraargs = {'isma': False, 'imdir': imdir, 'skeledges': skeledges, 'kpnames': kpnames, 'catname': 'fly'}
ConvertJson2COCO(infile,outfile,newimdir,extrainfo=extrainfo,**extraargs)

# %%
# # cp -r /groups/branson/home/robiea/.apt/tpfa9853c2_13f2_4007_aec1_7eeb2466c4e1/AddingWingsGTLabels/  /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_test
# # cp -r /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_test/im /groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/test
behaviors = {
    1: 'moving',
    2: 'grooming_or_standing',
    3: 'close',
    4: 'touching'
}

conditions = {
    1: 'courtship',
    2: 'female_aggression',
    3: 'male_aggression',
    4: 'non_social_lines',
    5: 'different_genotype',
    6: 'same_fly',
    7: 'same_genotype'    
}
label_category_file = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/label_categories.csv'

extrainfo['desription'] = 'Fly Bubble test data'
infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_test/loc.json'
outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/test_annotations.json'
newimdir = 'test'
testargs = {'behaviors': behaviors, 'conditions': conditions, 'label_category_file': label_category_file}
ConvertJson2COCO(infile,outfile,newimdir,extrainfo=extrainfo,**extraargs,**testargs)


# %%
infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_interlabeler/loc.json'
outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/test_inter_annotations.json'
newimdir = 'test_interlabeler'
ConvertJson2COCO(infile,outfile,newimdir,extrainfo=extrainfo,**extraargs)

# %%
infile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/cache_20241024_intralabeler/loc.json'
outfile = '/groups/branson/home/bransonk/tracking/code/APT/data/FlyBubble/fly_bubble_data_20241024/test_intra_annotations.json'
newimdir = 'test_intralabeler'
ConvertJson2COCO(infile,outfile,newimdir,extrainfo=extrainfo,**extraargs)

# %%
extraargs

# %%
import matplotlib.pyplot as plt

from pycocotools import coco
train_data = coco.COCO(outfile)
fig,ax = plt.subplots(3,3,figsize=(15,15))
ax = ax.flatten()
idx = np.random.choice(len(train_data.imgs),len(ax))
for axi,i in enumerate(idx):
    plt.sca(ax[axi])
    print(train_data.anns[i])
    print(train_data.imgs[i])
    imfile = os.path.join(imdir,train_data.imgs[i]['file_name'])
    plt.imshow(cv2.imread(imfile))
    train_data.showAnns([train_data.anns[i],])
    plt.gca().axis('image')
