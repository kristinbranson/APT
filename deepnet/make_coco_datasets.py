from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import re
plt.ion()

def describe_dataset(coco):

  for catid,cat in coco.cats.items():
    imgids = coco.getImgIds(catIds=catid)
    annids = coco.getAnnIds(catIds=catid)
    print(f"{catid}: {cat['name']}, {len(imgids)} images, {len(annids)} annotations, {len(cat['keypoints'])} keypoints.")

def show_examples(coco,imagedir):
  
  catids = []
  for catid in coco.cats.keys():
    annids = coco.getAnnIds(catIds=catid)
    if len(annids) > 0:
      catids.append(catid)
      
  ncats = len(catids)
  nc = int(np.ceil(np.sqrt(ncats)))
  nr = int(np.ceil(ncats/nc))
  fig,ax = plt.subplots(nc,nr,squeeze=False)
  fig.set_figheight(20)
  fig.set_figwidth(20)
  ax = ax.flatten()
  for i in range(ncats,len(ax)):
    plt.delaxes(ax[i])

  for i,catid in enumerate(catids):
    cat = coco.cats[catid]
    annids = coco.getAnnIds(catIds=catid)
    imgids = coco.getImgIds(catIds=catid)
    imgid = imgids[np.random.randint(len(imgids))]
    annIds = coco.getAnnIds(imgIds=imgid,catIds=catid)
    anns = coco.loadAnns(annIds)
    img = coco.loadImgs(imgid)[0]
    imgfile = os.path.join(imagedir,img['file_name'])
    if not os.path.exists(imgfile):
      print(f"{imgfile} does not exist")
      return
    I = cv2.imread(imgfile)
    I = I[...,::-1]
    ax[i].imshow(I)
    ax[i].axis('off')
    plt.sca(ax[i])
    coco.showAnns(anns)
    ax[i].set_title(f"{cat['name']}, {len(annids)}")
  fig.tight_layout()
  
def coco_select_video_subset(coco,video_ids):
  coco.dataset['images'] = list(filter(lambda img: img['video_id'] in video_ids,coco.dataset['images']))
  coco.dataset['annotations'] = list(filter(lambda ann: ann['video_id'] in video_ids,coco.dataset['annotations']))

def split_annotationfile_by_cat(annfile,debug=False):
  
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
  """
  buffalo: 2501
  lion: 2282
  dog: 2275
  horse: 2226
  deer: 2198
  rabbit: 2110
  cow: 2106
  zebra: 2025
  monkey: 1925
  pig: 1866
  giraffe: 1859
  panda: 1854
  cheetah: 1824
  polar-bear: 1810
  sheep: 1803
  elephant: 1795
  antelope: 1684
  wolf: 1670
  black-bear: 1638
  raccoon: 1589
  fox: 1521
  howler-monkey: 1497
  gorilla: 1496
  rhino: 1430
  hippo: 1405
  tiger: 1391
  cat: 1380
  spider-monkey: 1367
  orangutan: 1363
  chimpanzee: 1244
  """
  
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

  """
  dog: 794
  sheep: 355
  cat: 307
  antelope: 298
  zebra: 295
  bison: 268
  spider_monkey: 241
  hippo: 233
  cow: 228
  pig: 216
  giraffe: 210
  buffalo: 208
  deer: 206
  weasel: 199
  elephant: 193
  horse: 187
  jaguar: 187
  chimpanzee: 183
  wolf: 179
  rhino: 179
  lion: 177
  moose: 175
  rat: 175
  brown_bear: 171
  otter: 167
  panda: 164
  gorilla: 162
  monkey: 161
  hamster: 159
  fox: 157
  polar_bear: 156
  skunk: 155
  raccoon: 153
  rabbit: 153
  bobcat: 151
  mouse: 148
  cheetah: 148
  tiger: 144
  leopard: 142
  squirrel: 142
  beaver: 141
  argali_sheep: 110
  panther: 106
  noisy_night_monkey: 101
  marmot: 84
  snow_leopard: 73
  uakari: 69
  alouatta: 51
  black_bear: 39
  king_cheetah: 22
  """

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


if __name__ == "__main__":
  prepare_apt36k()

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# idxplot = 123

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
#   ax.plot(xsk,ysk,'-',color=colors[i%len(anns)])
#   ax.plot(x[v==1],y[v==1],'o',color=colors[i],ms=12,lw=4)
#   ax.plot(x[v==2],y[v==2],'+',color=colors[i],ms=12,lw=4)
  