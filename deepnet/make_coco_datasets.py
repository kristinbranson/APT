from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
plt.ion()

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def describe_dataset(coco):

  for catid,cat in coco.cats.items():
    imgids = coco.getImgIds(catIds=catid)
    annids = coco.getAnnIds(catIds=catid)
    print(f"{catid}: {cat['name']}, {len(imgids)} images, {len(annids)} annotations.")

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
    I = cv2.imread(imgfile)
    I = I[...,::-1]
    ax[i].imshow(I)
    ax[i].axis('off')
    plt.sca(ax[i])
    coco.showAnns(anns)
    ax[i].set_title(f"{cat['name']}, {len(annids)}")
  fig.tight_layout()

def split_annotationfile_by_cat(annfile):
  
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
    dataset['licenses'] = coco.dataset['licenses'].copy()
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
    with open(outannfile,'w') as outfid:
      outfid.write(json.dumps(dataset))

# coco
annfile = '/groups/branson/bransonlab/datasets/coco/annotations/person_keypoints_val2017.json'
imagedir = '/groups/branson/bransonlab/datasets/coco/images/val2017'

# ap10k
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

nann = np.zeros(len(coco.cats),dtype=int)
catnames = []
for i,catid in enumerate(coco.cats.keys()):
  nann[i] = len(coco.getAnnIds(catIds=catid))
  catnames.append(coco.cats[catid]['name'].replace(" ","_"))
order = np.argsort(-nann)
for i in order:
  print(f'{catnames[i]}: {nann[i]}')

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
  