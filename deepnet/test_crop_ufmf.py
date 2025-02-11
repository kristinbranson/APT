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
import crop_ufmf
from movies import Movie
import numpy as np
# make matplotlib plot inline
# %matplotlib inline
# make matplotlib only show errors
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2


# %%
# set parameters
inmoviefile = '/groups/branson/home/bransonk/tracking/code/APT/data/flyprism/fly_images/cam_0/image_cam_0_date_2024_12_20_time_16_05_44_v001.ufmf'
outdir = '/groups/branson/home/bransonk/tracking/code/APT/data/flyprism/cropped/cam_0'
cropcol = 1332
croprows = [[0,-1],[0,-1]]
cropcols = [[0,cropcol-1],[cropcol,-1]]
cropframes = [[0,199],[100,299]]
rot90 = 1
# croprows = [[0,-1]]
# cropcols = [[0,-1]]
# cropframes = [[0,100]]

# %%
# main command
outmoviefiles = crop_ufmf.cropufmf(inmoviefile,croprows=croprows,cropcols=cropcols,cropframes=cropframes,outdir=outdir,rot90=rot90)


# %%
# examine output ufmf index
outmoviefile = outmoviefiles[0]
testmovieobj = Movie(outmoviefile)
testmovieobj.open()
testindex = testmovieobj.h_mov._ufmf.get_index()
print(testindex)
testmovieobj.close()

# %%
# compare a frame
f = 150
inmovieobj = Movie(inmoviefile)

fig,ax = plt.subplots(3,len(outmoviefiles),figsize=(10*len(outmoviefiles),10),squeeze=False)
print(ax.shape)
inim,intimestamp = inmovieobj.get_frame(f)
for i in range(len(outmoviefiles)):
    if f < cropframes[i][0] or f > cropframes[i][1]:
        continue
    cf = f - cropframes[i][0]
    r0 = croprows[i][0]
    r1 = croprows[i][1]
    if r1 < 0:
        r1 = inim.shape[0] + r1
    c0 = cropcols[i][0]
    c1 = cropcols[i][1]
    if c1 < 0:
        c1 = inim.shape[1] + c1
    cropinim = inim[r0:r1+1,c0:c1+1]
    cropinim = np.rot90(cropinim,rot90)
    testmovieobj = Movie(outmoviefiles[i])
    testmovieobj.open()
    im,timestamp = testmovieobj.get_frame(cf)
    ax[0,i].imshow(cropinim,cmap='gray',vmin=0,vmax=255)
    ax[0,i].set_title(f'in {intimestamp}')
    ax[1,i].imshow(im,cmap='gray',vmin=0,vmax=255)
    ax[1,i].set_title(f'out {timestamp}')
    ax[2,i].imshow(cropinim-im)
    ax[2,i].set_title('diff')
    testmovieobj.close()    

inmovieobj.close()
