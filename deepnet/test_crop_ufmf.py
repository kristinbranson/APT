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

# main command
res = crop_ufmf.cropufmf(inmoviefile,croprows=croprows,cropcols=cropcols,cropframes=cropframes,outdir=outdir,rot90=rot90)
outmoviefiles = res['outmoviefiles']


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

fig,ax = plt.subplots(len(outmoviefiles),3,figsize=(30,10*len(outmoviefiles)),squeeze=False)
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
    ax[i,0].imshow(cropinim,cmap='gray',vmin=0,vmax=255)
    ax[i,0].set_title(f'in t = {intimestamp}')
    ax[i,1].imshow(im,cmap='gray',vmin=0,vmax=255)
    ax[i,1].set_title(f'out t = {timestamp}')
    him = ax[i,2].imshow(np.abs(cropinim-im))
    ax[i,2].set_title('absdiff, max = %f'%np.max(np.abs(cropinim-im)))
    # show colorbar with ax[i,2]
    plt.colorbar(him,ax=ax[i,2])
    testmovieobj.close()    
fig.tight_layout()
inmovieobj.close()

# %%
# for debugging what was going wrong on frame 150, column 1331

if False:

    i = 0
    f = 150
    inmovieobj = Movie(inmoviefile)
    inmovieobj.open()
    testmovieobj = Movie(outmoviefiles[i])
    testmovieobj.open()

    cf = f - cropframes[i][0]
    r0 = croprows[i][0]
    r1 = croprows[i][1]
    if r1 < 0:
        r1 = inim.shape[0] + r1
    c0 = cropcols[i][0]
    c1 = cropcols[i][1]
    if c1 < 0:
        c1 = inim.shape[1] + c1
    inim,intimestamp,inmore = inmovieobj.h_mov.get_frame(f,_return_more=True)
    cropinim = inim[r0:r1+1,c0:c1+1]
    cropinim = np.rot90(cropinim,rot90)

    im,timestamp,more = testmovieobj.h_mov.get_frame(cf,_return_more=True)
    ad = np.abs(cropinim-im)
    r,c = np.nonzero(ad)
    for i in range(len(r)):
        print(f'{r[i]},{c[i]}: in: {cropinim[r[i],c[i]]}, out: {im[r[i],c[i]]}, mean: {more["mean"][r[i],c[i]]}')

    fig,ax = plt.subplots(1,1,figsize=(30,10))
    plt.plot(cropinim[:,1331],'o',label='in')
    plt.plot(r,cropinim[r,1331],'o',label='in diff')
    #plt.plot(cropinim[1,:],label='in+1')
    plt.plot(im[:,1331],'x',label='out')
    plt.plot(r,im[r,1331],'x',label='out diff')
    plt.legend()

    testmovieobj.close()    
    inmovieobj.close()

    for region in inmore['regions']:
        x0 = region[0]
        y0 = region[1]
        data = region[2]
        x1 = x0 + data.shape[1]
        y1 = y0 + data.shape[0]
        if x0 <= 1331 and x1 >= 1331:
            minerr = np.inf
            for j in range(len(more['regions'])):
                outx0 = more['regions'][j][0]
                outy0 = more['regions'][j][1]
                outx1 = outx0 + more['regions'][j][2].shape[1]
                outy1 = outy0 + more['regions'][j][2].shape[0]
                err = np.abs(x0-outx0) + np.abs(x1-outx1) + np.abs(y0-outy0) + np.abs(y1-outy1)
                if err < minerr:
                    minerr = err
                    bestj = j
                    bestx0 = outx0
                    bestx1 = outx1
                    besty0 = outy0
                    besty1 = outy1
            print(f'x = {x0},{x1}, y = {y0},{y1}')
            print(f'closest to out region {bestj}: {bestx0},{bestx1}, {besty0},{besty1}')

# %%
inmoviefile = '/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/movie.ufmf'
cropframes = [10000,10099]
outmoviefiles = ['cx_GMR_SS00238_CsChr_RigC_20151007T150343_cropframes10000to19999.ufmf',]
res = crop_ufmf.cropufmf(inmoviefile,cropframes=cropframes,outmoviefiles=outmoviefiles)
outmoviefiles = res['outmoviefiles']
croprows = res['croprows']
cropcols = res['cropcols']
cropframes = res['cropframes']
outdir = res['outdir']
rot90 = res['rot90']

