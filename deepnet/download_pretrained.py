import os
import tarfile
import urllib
url = 'http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz'
script_dir = os.path.dirname(os.path.realpath(__file__))
wt_dir = os.path.join(script_dir,'pretrained')
wt_file = os.path.join(wt_dir,'resnet_v2_fp32_savedmodel_NHWC','1538687283','variables','variables.index')
if not os.path.exists(wt_file):
    print('Downloading slim resnet pretrained weights..')
    if not os.path.exists(wt_dir):
        os.makedirs(wt_dir)
    sname, header = urllib.urlretrieve(url)
    tar = tarfile.open(sname, "r:gz")
    print('Extracting pretrained weights..')
    tar.extractall(path=wt_dir)
url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
script_dir = os.path.dirname(os.path.realpath(__file__))
wt_dir = os.path.join(script_dir,'pretrained')
wt_file = os.path.join(wt_dir,'resnet_v1_50.ckpt')
if not os.path.exists(wt_file):
    print('Downloading tensorflow resnet pretrained weights..')
    if not os.path.exists(wt_dir):
        os.makedirs(wt_dir)
    sname, header = urllib.urlretrieve(url)
    tar = tarfile.open(sname, "r:gz")
    print('Extracting pretrained weights..')
    tar.extractall(path=wt_dir)
