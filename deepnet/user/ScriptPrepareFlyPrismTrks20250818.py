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
#     display_name: APT
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import TrkFile
import os
import sys
import re
import matplotlib.pyplot as plt


# %%
# data locations
rootdatadir = '/groups/branson/bransonlab/aniket/APT/3D_labeling_project/movie_output_dir_combined_views'
rootoutdir = '/groups/branson/home/bransonk/tracking/code/APT/debug/flyprism_merged_trks'

if not os.path.exists(rootoutdir):
    os.makedirs(rootoutdir)

# parameters
viewtypes = ['Bottom','Side']
nviewspertype = 2
nviews = len(viewtypes) * nviewspertype
nettype = 'magrone'
minconf = 0.5

# %%
# get full paths to subdirectories in rootdatadir named exp*
expdirs = [os.path.join(rootdatadir, d) for d in os.listdir(rootdatadir) if d.startswith('exp') and os.path.isdir(os.path.join(rootdatadir, d))]
print(f"Found {len(expdirs)} experiment directories: {expdirs}")
nexps = len(expdirs)

trkfiles = {}
found_trkfile = np.zeros((nexps,len(viewtypes),nviewspertype), dtype=bool)

for expdir in expdirs:
    trkfiles[expdir] = {}
    for viewtype in viewtypes:
        for viewi in range(nviewspertype):
            # find trk file: trk file name will be {expdir}/image_cam_{view}*combined{viewtype}View*_{nettype}*.trk, and does not contain tracklet, use regex to match
            files = [f for f in os.listdir(expdir) if re.match(rf'image_cam_{viewi}.*combined{viewtype}View.*_{nettype}.*\.trk', f) and not 'tracklet' in f]
            if len(files) == 0:
                print(f"No trkfiles found for {expdir}, viewtype {viewtype}, viewi {viewi}")
                continue
            elif len(files) > 1:
                print(f"Multiple trkfiles found for {expdir}, viewtype {viewtype}, viewi {viewi}: {files}")
                continue
            else:
                trkfile = os.path.join(expdir, files[0])
                trkfiles[expdir][(viewtype, viewi)] = trkfile
                found_trkfile[expdirs.index(expdir), viewtypes.index(viewtype), viewi] = True


# %%
def merge_tracklets(trkfiles, minconf=minconf):
    """
    Merge tracklets for each trkfile into a single tracklet so that it can be read into multiview project
    """

    trkins = {}
    for (viewtype, viewi), trkfile in trkfiles.items():
        trkins[(viewtype,viewi)] = TrkFile.Trk(trkfile)
    allT1 = np.max([trkin.T1 for trkin in trkins.values()])
    alltracked = np.ones(allT1+1, dtype=bool)
    for trkin in trkins.values():
        istracked = np.zeros(trkin.T1+1, dtype=bool)
        for tgt in range(trkin.ntargets):
            p = trkin.gettarget(tgt)
            isreal = np.all(np.isnan(p), axis=(0, 1, 3)) == False
            istracked[isreal] = True
        istracked = np.r_[istracked, np.zeros(allT1 - trkin.T1, dtype=bool)]
        alltracked = np.logical_and(alltracked, istracked)

    trkouts = {}
    
    for (viewtype, viewi), trkin in trkins.items():

        nkpts = trkin.nlandmarks
        d = trkin.d
        ntargets = trkin.ntargets
        T1 = trkin.T1
        
        # only set data for frames that are tracked in all views
        isset = alltracked[:T1+1] == False
        p = np.zeros((nkpts, d, allT1+1,1))
        p[:] = np.nan
        rest = {}

        # do not deal with multiple targets on the same frame, just choose one
        # do this in order of targets based on max confidence
        maxconf = np.zeros(ntargets)
        for tgt in range(ntargets):
            pcurr, restcurr = trkin.gettarget(tgt, extra=True)
            maxconf[tgt] = np.nanmax(restcurr['pTrkConf'])
            if np.isnan(maxconf[tgt]):
                #print(f"Target {tgt} has all nan confidence")
                maxconf[tgt] = minconf

        ordered_targets = np.argsort(maxconf)[::-1]  # sort targets by max confidence, descending

        for tgt in ordered_targets:
            pcurr,restcurr = trkin.gettarget(tgt,extra=True)
            isreal = np.all(np.isnan(pcurr),axis=(0,1,3)) == False
            toset = np.logical_and(isreal, ~isset)
            #print(f"Target {tgt} has {np.sum(toset)} frames with conf >= {minconf} and not already set")
            if not np.any(toset):
                continue
            toset = np.nonzero(toset)[0]
            isset[toset] = True
            p[:, :, toset,:] = pcurr[:, :, toset,:]
            for k in restcurr.keys():
                if restcurr[k] is None:
                    continue
                if k not in rest:
                    szcurr = restcurr[k].shape
                    sz = szcurr[:-2] + (allT1+1,) + szcurr[-1:]
                    rest[k] = np.zeros(sz,dtype=type(trkin.defaultval_dict[k]))
                    rest[k][:] = trkin.defaultval_dict[k]
                rest[k][..., toset,:] = restcurr[k][..., toset,:]
                
        trkout = TrkFile.Trk(p=p,**rest)
        trkout.convert2sparse()
        trkouts[(viewtype, viewi)] = trkout

    return trkouts


# %%
for expi in range(nexps):
    expdir = expdirs[expi]

    trkfilescurr = trkfiles[expdir]
    # check if all views have a trk file
    if not np.all(found_trkfile[expi, :, :]):
        print(f"Not all views have a trk file for {expdir}. Skipping this experiment.")
        continue
    
    trkouts = merge_tracklets(trkfilescurr, minconf=minconf)
    for viewtype in viewtypes:
        for viewi in range(nviewspertype):
            trkoutfile = os.path.join(rootoutdir, f"{os.path.basename(expdir)}_combined{viewtype}View_cam{viewi}.trk")
            print(f"Saving merged tracklets to {trkoutfile}")
            trkout = trkouts[(viewtype, viewi)]
            trkout.save(trkoutfile)


# %%
for view in range(nviewspertype):
    print(f"cam {view}: T0 = {trkouts[('Bottom',view)].T0}, T1 = {trkouts[('Bottom',view)].T1}")

# %%
expdirs

# %%
expi = 7
fig,ax = plt.subplots(3,1, figsize=(15, 6), sharex=True)
kpi = 0

for viewtype in viewtypes:
    for viewi in range(nviewspertype):
        outtrkfile = os.path.join(rootoutdir, f"{os.path.basename(expdirs[expi])}_combined{viewtype}View_cam{viewi}.trk")
        trkout = TrkFile.Trk(trkfile=outtrkfile)
        
        p,rest = trkout.gettarget(0, extra=True)
        l = f'{viewtype} cam {viewi}'
        ax[0].plot(rest['pTrkConf'].max(axis=0),'.',label=l)
        ax[0].set_ylabel('max confidence')
        ax[1].plot(p[kpi,0,:],'.',label=l)
        ax[1].set_ylabel(f'kpt {kpi} x-coord')
        ax[2].plot(p[kpi,1,:],'.',label=l)
        ax[2].set_ylabel(f'kpt {kpi} y-coord')

ax[0].legend()

