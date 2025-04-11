from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import re
from deeplabcut.utils import auxiliaryfunctions as dlc_aux
import deeplabcut as dlc
import pandas as pd
import pickle

plt.ion()

def describe_dataset(coco):

  for catid,cat in coco.cats.items():
    imgids = coco.getImgIds(catIds=catid)
    annids = coco.getAnnIds(catIds=catid)
    print(f"{catid}: {cat['name']}, {len(imgids)} images, {len(annids)} annotations, {len(cat['keypoints'])} keypoints.")

def show_examples(coco,imagedir,nexamples=1):
  
  catids = []
  for catid in coco.cats.keys():
    annids = coco.getAnnIds(catIds=catid)
    if len(annids) > 0:
      catids.append(catid)
      
  ncats = len(catids)
  
  nax = nexamples*ncats
  nc = int(np.ceil(np.sqrt(nax)))
  nr = int(np.ceil(nax/nc))
  fig,ax = plt.subplots(nc,nr,squeeze=False)
  fig.set_figheight(20)
  fig.set_figwidth(20)
  ax = ax.flatten()

  axi = 0
  for i,catid in enumerate(catids):
    cat = coco.cats[catid]
    annids = coco.getAnnIds(catIds=catid)
    imgids = coco.getImgIds(catIds=catid)
    idx = np.random.choice(len(imgids),np.minimum(nexamples,len(imgids)),replace=False)
    for j in idx:
      imgid = imgids[j]
      annIds = coco.getAnnIds(imgIds=imgid,catIds=catid)
      anns = coco.loadAnns(annIds)
      img = coco.loadImgs(imgid)[0]
      imgfile = os.path.join(imagedir,img['file_name'])
      if not os.path.exists(imgfile):
        print(f"{imgfile} does not exist")
        return
      I = cv2.imread(imgfile)
      I = I[...,::-1]
      ax[axi].imshow(I)
      ax[axi].axis('off')
      plt.sca(ax[axi])
      coco.showAnns(anns)
      ax[axi].set_title(f"{cat['name']}, {len(annids)}: {imgid}")
      axi+=1

  for i in range(axi,len(ax)):
    plt.delaxes(ax[i])

  fig.tight_layout()
  
def coco_select_video_subset(coco,video_ids):
  coco.dataset['images'] = list(filter(lambda img: img['video_id'] in video_ids,coco.dataset['images']))
  coco.dataset['annotations'] = list(filter(lambda ann: ann['video_id'] in video_ids,coco.dataset['annotations']))

def split_annotationfile_by_cat(annfile,debug=False):
  
  outannfiles = {}
  coco = COCO(annfile)
  p,n = os.path.split(annfile)
  n,ext = os.path.splitext(n)
  for catid in coco.cats.keys():
    annids = coco.getAnnIds(catIds=catid)
    if len(annids) == 0:
      continue
    cat = coco.cats[catid]
    catname = cat['name'].replace(" ","_")
    outannfile = os.path.join(p,f"{n}-{catname}{ext}")
    dataset = {}

    # info
    dataset['info'] = coco.dataset['info'].copy()
    dataset['info']['description'] = f"{coco.dataset['info']['description']}-{cat['name']}"
    # licenses
    if 'licenses' in dataset:
      dataset['licenses'] = coco.dataset['licenses'].copy()
    elif 'license' in dataset:
      dataset['license'] = coco.dataset['license'].copy()
    # images
    imgids = coco.getImgIds(catIds=catid)
    imgs = coco.loadImgs(imgids)
    dataset['images'] = imgs
    # annotations
    annids = coco.getAnnIds(catIds=catid)
    anns = coco.loadAnns(annids)
    dataset['annotations'] = anns
    # categories
    dataset['categories'] = [cat,]

    print(f'creating ann file {outannfile}')
    if not debug:
      with open(outannfile,'w') as outfid:
        outfid.write(json.dumps(dataset))
    outannfiles[catname] = outannfile
  return outannfiles

def print_nannotations(coco):

  nann = np.zeros(len(coco.cats),dtype=int)
  catnames = []
  for i,catid in enumerate(coco.cats.keys()):
    nann[i] = len(coco.getAnnIds(catIds=catid))
    catnames.append(coco.cats[catid]['name'].replace(" ","_"))
  order = np.argsort(-nann)
  for i in order:
    print(f'{catnames[i]}: {nann[i]}')

  return nann,catnames

# # coco
# annfile = '/groups/branson/bransonlab/datasets/coco/annotations/person_keypoints_val2017.json'
# imagedir = '/groups/branson/bransonlab/datasets/coco/images/val2017'

def prepare_apt36k():

  basedir = '/groups/branson/bransonlab/datasets/APT-36K'
  annfile = os.path.join(basedir,'annotations','apt36k_annotations.json')
  imagedir = os.path.join(basedir,'data')
  coco = COCO(annfile)
  describe_dataset(coco)
  fixedannfile = os.path.join(basedir,'annotations','apt36k_annotations_fixed.json')
  
  # fix image names
  for i,img in enumerate(coco.dataset['images']):
    img['file_name'] =  img['file_name'].replace("D:\\Animal_pose\\AP-36k-patr1\\","").replace("\\","/")
    if not os.path.exists(img['file_name']):
      if re.search('howling-monkey',img['file_name']) is not None:
        img['file_name'] =  img['file_name'].replace('howling-monkey','howler-monkey')
      elif re.search('raccon',img['file_name']) is not None:
        img['file_name'] =  img['file_name'].replace('raccon','raccoon')
      elif re.search('hourse',img['file_name']) is not None:
        img['file_name'] =  img['file_name'].replace('hourse','horse')
      elif re.search('1deer/clip46',img['file_name']) is not None:
        p,n = os.path.split(img['file_name'])
        img['file_name'] = os.path.join('1deer/clip46',n)
      elif re.search('29wolf/video11_clip6_1m44s-1m46s_frame',img['file_name']) is not None:
        p,n = os.path.split(img['file_name'])
        img['file_name'] = os.path.join('29wolf/video11_clip6_1m44s-1m46s_frame',n)
      elif re.search('29wolf/video20_clip5_1m44s-1m46s_frame',img['file_name']) is not None:
        p,n = os.path.split(img['file_name'])
        img['file_name'] = os.path.join('29wolf/video20_clip5_1m44s-1m46s_frame',n)
      elif re.search('29wolf/video2_clip3_0m55s-0m57s_frame',img['file_name']) is not None:
        p,n = os.path.split(img['file_name'])
        img['file_name'] = os.path.join('29wolf/video2_clip3_0m55s-0m57s_frame',n)
      elif re.search('6pig/v2c21',img['file_name']) is not None:
        p,n = os.path.split(img['file_name'])
        img['file_name'] = os.path.join('6pig/v2c21',n)
      
    assert os.path.exists(os.path.join(imagedir,img['file_name']))
  
    for i,cat in enumerate(coco.cats.values()):
      if re.search('howling-monkey',cat['name']) is not None:
        cat['name'] =  cat['name'].replace('howling-monkey','howler-monkey')
      elif re.search('raccon',cat['name']) is not None:
        cat['name'] =  cat['name'].replace('raccon','raccoon')
      elif re.search('hourse',cat['name']) is not None:
        cat['name'] =  cat['name'].replace('hourse','horse')
        
    with open(fixedannfile,'w') as outfid:
      outfid.write(json.dumps(coco.dataset))
  
  show_examples(coco,imagedir)
  split_annotationfile_by_cat(fixedannfile)
  print_nannotations(coco)
  
  # split into train, val, test  
  
  # make sure each video only contains one animal type
  all_video_ids = np.unique(np.array([img['video_id'] for img in coco.dataset['images']]))
  nannpervid = np.zeros((np.max(all_video_ids)+1,len(coco.cats)),dtype=int)
  for ann in coco.dataset['annotations']:
    nannpervid[ann['video_id'],ann['category_id']-1] += 1
  assert np.max(np.sum(nannpervid>0,axis=1)) == 1
  
  splitfrac = {'train': .7, 'val': .1, 'test': .2}
  videos_split = {}
  for k in splitfrac.keys():
    videos_split[k] = np.array([])
  
  for i,catid in enumerate(coco.cats.keys()):
    imgids = coco.getImgIds(catIds=catid)
    imgs = coco.loadImgs(imgids)
    video_ids = np.array([img['video_id'] for img in imgs])
    unique_video_ids,video_idx = np.unique(video_ids,return_inverse=True)
    nvideos = len(unique_video_ids)
    nvideosleft = nvideos
    nvideos_split = {}
    for k,v in splitfrac.items():
      fleft = nvideosleft/nvideos
      nvideoscurr = int(np.round(nvideosleft*(v/fleft)))
      nvideos_split[k] = nvideoscurr
      nvideosleft -= nvideoscurr
    video_order = unique_video_ids[np.random.permutation(nvideos)]
    off = 0
    for k,v in nvideos_split.items():
      videos_split[k] = np.r_[videos_split[k],video_order[off:off+v]]
      off += v

  p,n = os.path.split(fixedannfile)
  n,ext = os.path.splitext(n)
  for k,v in videos_split.items():
    cococurr = COCO(fixedannfile)
    coco_select_video_subset(cococurr,v)
    outannfile = os.path.join(p,f"{n}_{k}{ext}")
    with open(outannfile,'w') as outfid:
      outfid.write(json.dumps(cococurr.dataset))
    cococurr = COCO(outannfile)
    show_examples(cococurr,imagedir)
    split_annotationfile_by_cat(outannfile)

  for cat in coco.cats.values():
    catname = cat['name']
    print(f'{catname}: ')
    for split in splitfrac.keys():
      annfilecurr = os.path.join(p,f"{n}_{split}-{catname}{ext}")
      dataset = json.load(open(annfilecurr,'r'))
      nann = len(dataset['annotations'])
      nvid = len(np.unique(np.array([img['video_id'] for img in dataset['images']])))
      print(f'  {split}: {nann} annotations, {nvid} videos')

def prepare_ap10k():

  basedir = '/groups/branson/bransonlab/datasets/ap-10k'
  files = ['ap10k-train-split1.json','ap10k-val-split1.json','ap10k-test-split1.json']
  annfiles = [os.path.join(basedir,'annotations',x) for x in files]
  imagedirs = [os.path.join(basedir,'data'),]*len(annfiles)

  for i in range(len(annfiles)):
    annfile = annfiles[i]
    imagedir = imagedirs[i]
    assert os.path.exists(annfile)
    assert os.path.exists(imagedir)

    coco = COCO(annfile)
    describe_dataset(coco)
    show_examples(coco,imagedir)
    plt.show()
    
    # make an ann file per category
    split_annotationfile_by_cat(annfile)


  # list categories by number of train examples
  annfile = annfiles[0]
  coco = COCO(annfile)
  print_nannotations(coco)
  
def prepare_deeplabcut(basedir,categories,info,trainlistfile=None):
    
  configfile = os.path.join(basedir,'config.yaml')
  cfg = dlc_aux.read_config(configfile)
  
  outannfiles = {}
  
  # with open(os.path.join(splitfile),'rb') as fid:
  #   splitinfo = pickle.load(fid)
  
  #dlc.create_training_dataset(configfile)

  bodyparts = list(cfg['bodyparts'])
  skeleton = list(cfg['skeleton'])
  try:
    skeleton_idx = [[bodyparts.index(s[0])+1,bodyparts.index(s[1])+1] for s in skeleton]
  except:
    skeleton_idx = None
  
  outannfile = os.path.join(basedir,'annotations.json')
  outannfiles['all'] = outannfile
  
  videos = cfg["video_sets"].keys()
  video_names = []
  folders = []
  for video in videos:
    n,ext = os.path.splitext(os.path.basename(video))
    video_names.append(n)
    folders.append(os.path.join(cfg['project_path'],'labeled-data',n))

  dataset = {}
  if type(categories) == str:
    categories = {categories: None}
  
  dataset['categories'] = []
  catid = 1
  for category in categories.keys():
    cat = {
      'id': catid,
      'name': category,
      'keypoints': None,
      'skeleton': None,
    }
    dataset['categories'].append(cat)
    catid += 1
    
  dataset['images'] = []
  dataset['annotations'] = []
  dataset['licenses'] = [
    {
      'id': 1,
      'name': 'Creative Commons Attribution 4.0 International',
      'url': 'https://creativecommons.org/licenses/by/4.0/legalcode',
    },
  ]
  dataset['info'] = info
  
  annid = 1
  imgid = 1
  id = 1
    
  for videoid in range(len(video_names)):
    folder = folders[videoid]
    video_name = video_names[videoid]
    print(f'video {videoid} / {len(video_names)}: {video_name}')
    DataCombined = pd.read_hdf(os.path.join(str(folder), "CollectedData_" + cfg["scorer"] + ".h5"))
    imgnames = DataCombined.index
    annnames = DataCombined.columns

    anninfo = {}
    for i,n in enumerate(annnames.names):
      anninfo[n] = [a[i] for a in annnames]

    ismulti = 'individuals' in anninfo

    if ismulti:
      individuals = list(set(anninfo['individuals']))
    else:
      individuals = ['dummy',]

    individ2catidx = {}
    for individual in individuals:
      if ismulti:
        idx = (cfg['scorer'],individual)
      else:
        idx = cfg['scorer']
      bodyparts = list(set([x[0] for x in DataCombined.loc[:,idx]]))

      for catidx,catname in enumerate(categories.keys()):
        if (categories[catname] is None) or (individual in categories[catname]):
          individ2catidx[individual] = catidx
          break
      assert individual in individ2catidx
      if dataset['categories'][catidx]['keypoints'] is None:
        skeleton_idx = []
        for s in skeleton:
          if (s[0] in bodyparts) and (s[1] in bodyparts):
            skeleton_idx.append([bodyparts.index(s[0])+1,bodyparts.index(s[1])+1])
        dataset['categories'][catidx]['keypoints'] = bodyparts
        dataset['categories'][catidx]['skeleton'] = skeleton_idx
  
      assert set(dataset['categories'][catidx]['keypoints']) == set(bodyparts)

    width = None
    height = None
    for imgname in imgnames:
      if imgid == 1:
        I = cv2.imread(os.path.join(basedir,imgname))
        width = I.shape[0]
        height = I.shape[1]
        
      img = {
        'license': 1,
        'id': imgid,
        'file_name': imgname,
        'video_id': videoid,
        'width': width,
        'height': height,
      }
      dataset['images'].append(img)

      for individual in individuals:

        catidx = individ2catidx[individual]
        bodyparts = dataset['categories'][catidx]['keypoints']        
        
        x = np.zeros(len(bodyparts))
        y = np.zeros(len(bodyparts))
        v = np.ones(len(bodyparts))
        if ismulti:
          idx = (cfg['scorer'],individual)
        else:
          idx = (cfg['scorer'],)
        for i,bodypart in enumerate(bodyparts):
          if (idx+(bodypart,'x')) in DataCombined.columns:
            x[i] = DataCombined.loc[imgname,idx+(bodypart,'x')]
            y[i] = DataCombined.loc[imgname,idx+(bodypart,'y')]
          else:
            x[i] = np.nan
            y[i] = np.nan
            
        v = ((np.isnan(x) | np.isnan(y)) == False).astype(int)*2
        x[v==0] = 0.
        y[v==0] = 0.
        keypoints = np.c_[x,y,v].flatten()
        ann = {
          'id': annid,
          'keypoints': keypoints.tolist(),
          'track_id': id,
          'image_id': imgid,
          'category_id': dataset['categories'][catidx]['id'],
          'video_id': videoid,
        }
        dataset['annotations'].append(ann)
        annid += 1
        id += 1
      imgid+=1

  with open(outannfile,'w') as outfid:
    outfid.write(json.dumps(dataset))
  coco = COCO(outannfile)

  # assert len(dataset['images']) == len(splitinfo[0])
  # all_file_names = [img['file_name'] for img in dataset['images']]
  # imagemapping = np.zeros(len(dataset['images']),dtype=int)
  # for i,img in enumerate(splitinfo[0]):
  #   idx = all_file_names.index(img['image'])
  #   assert idx is not None
  #   imagemapping[i] = idx
  # trainidx = imagemapping[splitinfo[1]]
  
  show_examples(coco,basedir,nexamples=int(np.round(16/len(coco.cats))))
  
  if len(categories) > 1:
    outannfiles['category'] = split_annotationfile_by_cat(outannfile)
  
  if trainlistfile is not None:
    with open(os.path.join(basedir,trainlistfile),'rb') as fid:
      splitinfo = pickle.load(fid)
    trainimages = splitinfo[0]
    traindirs = set([os.path.basename(os.path.dirname(x['image'])) for x in trainimages])
    trainidx = [video_names.index(x) for x in traindirs]
    coco_train = COCO(outannfile)
    coco_select_video_subset(coco_train,trainidx)
    trainannfile = os.path.join(basedir,'annotations_train.json')
    with open(trainannfile,'w') as outfid:
      outfid.write(json.dumps(dataset))
    outannfiles['train'] = trainannfile
    coco_train = COCO(trainannfile)
    show_examples(coco_train,basedir,nexamples=9)
    testidx = list(set(range(len(video_names))).difference(trainidx))
    coco_test = COCO(outannfile)
    coco_select_video_subset(coco_test,testidx)
    testannfile = os.path.join(basedir,'annotations_test.json')
    with open(testannfile,'w') as outfid:
      outfid.write(json.dumps(dataset))
    coco_test = COCO(testannfile)
    outannfiles['test'] = testannfile
    show_examples(coco_train,basedir,nexamples=9)

  return outannfiles

def prepare_deeplabcut_all(datasets2do=None):
    
  rootdir = '/groups/branson/bransonlab/datasets/deeplabcut'
      
  basedirs = {
    'mouse': os.path.join(rootdir,'trimice-dlc-2021-06-22'),
    'pups': os.path.join(rootdir,'pups-dlc-2021-03-24'),
    'fish': os.path.join(rootdir,'fish-dlc-2021-05-07'),
    'marmoset': os.path.join(rootdir,'marmoset-dlc-2021-05-07'),
    'horse': os.path.join(rootdir,'horse10'),
  }
  categories = {
    'mouse': 'mouse',
    'pups': {
        'mouse': ['single',],
        'pup': ['p1','p2'],
      },
    'fish': 'fish',
    'marmoset': 'marmoset',
    'horse': 'horse',
  }    
  infos = {
    'mouse': {
      'description': 'maDLC Tri-Mouse Benchmark by Lauer ... Mathis, converted to COCO by Branson 20230523',
      'url': 'https://zenodo.org/record/5851157#.YeHC23vMJhE',
    },
    'pups': {
      'description': 'maDLC Parenting Benchmark Dataset by Lauer ... Mathis, converted to COCO by Branson 20230523',
      'url': 'https://zenodo.org/record/5851109#.YeHC3nvMJhE',
    },
    'fish': {
      'description': 'maDLC Fish Benchmark Dataset by Lauer ... Mathis, converted to COCO by Branson 20230523',
      'url': 'https://zenodo.org/record/5849286#.YeHC4XvMJhE',
    },
    'marmoset': {
      'description': 'maDLC Marmoset Benchmark Dataset by Lauer ... Mathis, converted to COCO by Branson 20230523',
      'url': 'https://zenodo.org/record/5849371#.YeHC3nvMJhE',
    },
    'horse': {
      'description': 'Horse-10 by Rogers ... Mathism converted to COCO by Branson 20230523',
      'url': 'http://www.mackenziemathislab.org/horse10',
    }
  }
  
  trainlistfiles = {
    'horse': 'training-datasets/iteration-0/UnaugmentedDataSet_HorsesMay8/Documentation_data-Horses_50shuffle1.pickle',
  }
  
  alldatasets = list(basedirs.keys())
  if datasets2do is None:
    datasets2do = alldatasets
    
  outannfiles = {}
  for dataset in datasets2do:
    if dataset in trainlistfiles:
      trainlistfile = trainlistfiles[dataset]
    else:
      trainlistfile = None
    outannfiles[dataset] = prepare_deeplabcut(basedirs[dataset],categories[dataset],infos[dataset],trainlistfile)

  for dataset in alldatasets:
    annfile = os.path.join(basedirs[dataset],'annotations.json')
    coco = COCO(annfile)
    print(f'{dataset}:')
    describe_dataset(coco)

    annfile = os.path.join(basedirs[dataset],'annotations_train.json')
    if os.path.exists(annfile):
      coco = COCO(annfile)
      print(f'{dataset}_train:')
      describe_dataset(coco)

    annfile = os.path.join(basedirs[dataset],'annotations_test.json')
    if os.path.exists(annfile):
      coco = COCO(annfile)
      print(f'{dataset}_test:')
      describe_dataset(coco)


  return

if __name__ == "__main__":
  #prepare_ap10k()
  #prepare_apt36k()
  datasets2do = [] # None #['horse',]
  prepare_deeplabcut_all(datasets2do)


# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# idxplot = 1
# imagedir = basedir

# imgids = coco.getImgIds()
# img = coco.loadImgs(imgids[idxplot])[0]
# imgfile = os.path.join(imagedir,img['file_name'])
# assert os.path.exists(imgfile)
# annIds = coco.getAnnIds(imgIds=img['id'])
# anns = coco.loadAnns(annIds)
# I = cv2.imread(imgfile)
# I = I[...,::-1]

# plt.clf()
# plt.imshow(I)
# ax = plt.gca()
# for i,ann in enumerate(anns):
#   x = np.array(ann['keypoints'][0::3]).astype(np.float32)
#   y = np.array(ann['keypoints'][1::3]).astype(np.float32)
#   v = np.array(ann['keypoints'][2::3])
#   x[v==0] = np.nan
#   y[v==0] = np.nan
#   sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
#   xsk = np.c_[x[sks],np.zeros(len(sks))+np.nan].T
#   ysk = np.c_[y[sks],np.zeros(len(sks))+np.nan].T
#   #ax.plot(xsk,ysk,'-',color=colors[i%len(anns)])
#   ax.plot(x[v==1],y[v==1],'o',color=colors[i],ms=12,lw=4)
#   ax.plot(x[v==2],y[v==2],'+',color=colors[i],ms=12,lw=4)
#   keypointnames = coco.loadCats(ann['category_id'])[0]['keypoints']
#   for j,kp in enumerate(keypointnames):
#     ax.text(x[j],y[j],kp,color='')
  