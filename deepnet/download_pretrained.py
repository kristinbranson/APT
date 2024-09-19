#! /usr/bin/env python3

import os
import tarfile
import urllib.request as urllib

def download_and_extract(url,script_dir,wt_dir,wt_file,wt_file_gz,name=''):

    if os.path.exists(wt_file):
        return

    print('Downloading %s pretrained weights..'%name)
    if not os.path.exists(wt_dir):
        os.makedirs(wt_dir)
        assert os.path.exists(wt_dir), 'Directory %s could not be created'%wt_dir
    sname, header = urllib.urlretrieve(url,wt_file_gz)
    assert os.path.exists(sname), 'Could not download from %s to %s'%(url,sname)
    tar = tarfile.open(sname, "r:gz")
    print('Extracting pretrained weights..')
    tar.extractall(path=wt_dir)
    os.remove(wt_file_gz)

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.realpath(__file__))
    wt_dir = os.path.join(script_dir,'pretrained')

    url = 'http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC.tar.gz'
    wt_file = os.path.join(wt_dir,'resnet_v2_fp32_savedmodel_NHWC','1538687283','variables','variables.index')
    wt_file_gz = os.path.join(wt_dir,'resnet_v2_fp32_savedmodel_NHWC.tar.gz')
    name = 'slim resnet'

    download_and_extract(url,script_dir,wt_dir,wt_file,wt_file_gz,name)

    url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz'
    wt_file = os.path.join(wt_dir,'resnet_v1_50.ckpt')
    wt_file_gz = os.path.join(wt_dir,'resnet_v1_50_2016_08_28.tar.gz')
    name = 'tensorflow resnet'
    download_and_extract(url,script_dir,wt_dir,wt_file,wt_file_gz,name)
