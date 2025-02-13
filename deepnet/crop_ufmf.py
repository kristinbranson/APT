from movies import Movie
from ufmf import UfmfSaver
import os
import tqdm
import argparse
import ast
import numpy as np

ufmf_ver = 3

def rotate_region(x,y,w,h,maxwidth,maxheight,rot90):
    """
    rotate_region(x,y,w,h,maxwidth,maxheight,rot90)
    Rotate a region by rot90*90 degree counter-clockwise
    x,y: top-left corner of the region
    w,h: width and height of the region
    maxwidth,maxheight: maximum width and height of the frame
    rot90: number of 90 degree counter-clockwise rotations (0, 1, 2, or 3)
    Notes:
    Did this manually in my head, should probably check... 
    Outputs:
    x,y: top-left corner of the rotated region
    w,h: width and height of the rotated region
    """
    rot90 = rot90 % 4
    if rot90 == 0:
        return x, y, w, h
    elif rot90 == 1:
        return y, maxwidth-x-w, h, w
    elif rot90 == 2:
        return maxwidth-x-h, maxheight-y-w, w, h
    elif rot90 == 3:
        return maxheight-y-h, x, h, w
    else:
        raise ValueError('rot90 must be 0, 1, 2, or 3')

def cropframe(frame,more,croprows,cropcols):
    """
    cropframe(frame,more,croprows,cropcols)
    Crop a frame represented with regions
    frame: the frame, np.ndarray of size maxheight,maxwidth,...
    more: regions in the frame, dict with key 'regions'
    croprows: [startrow,endrow] to crop
    cropcols: [startcol,endcol] to crop
    Returns:
    cfr: cropped frame
    cropregions: cropped regions, list of [x,y,w,h,data] where x,y are top-left corner of the region,
    w,h are width and height of the region, and data is the cropped region
    """
    cropregions = []
    for region in more['regions']:
        x = region[0]
        y = region[1]
        data = region[2]
        if x > cropcols[1] or x + data.shape[1] <= cropcols[0] or \
            y > croprows[1] or y + data.shape[0] <= croprows[0]:
            continue
        cropdata = data.copy()
        cropx = x - cropcols[0]
        cropy = y - croprows[0]
        if x < cropcols[0]:
            cropdata = cropdata[:,cropcols[0]-x:]
            cropx = 0
        if x + data.shape[1] - 1 > cropcols[1]:
            x1 = (x+data.shape[1])-cropcols[1] - 1
            assert x1 > 0
            cropdata = cropdata[:,:-x1]
        if y < croprows[0]:
            cropdata = cropdata[croprows[0]-y:,:]
            cropy = 0
        if y + data.shape[0] - 1 > croprows[1]:
            y1 = (y+data.shape[0]) - croprows[1] - 1
            assert y1 > 0
            cropdata = cropdata[:-y1,:]
        w = cropdata.shape[1]
        h = cropdata.shape[0]
        assert w > 0 and h > 0
        cropregions.append([cropx,cropy,w,h,cropdata])
    cfr = frame[croprows[0]:croprows[1]+1,cropcols[0]:cropcols[1]+1]
    return cfr,cropregions

def rotateframe(frame,regions,rot90=0):
    """
    rotateframe(frame,regions,rot90=0)
    Rotate a frame and its regions by rot90*90 degree counter-clockwise
    frame: the frame, np.ndarray of size maxheight,maxwidth,...
    regions: regions in the frame, list of [x,y,w,h,data] where x,y are top-left corner of the region,
    w,h are width and height of the region, and data is the region
    rot90: integer, number of 90 degree counter-clockwise rotations
    Returns:
    rotframe: rotated frame
    rotregions: rotated regions, list of [x,y,w,h,data] where x,y are top-left corner of the region,
    w,h are width and height of the region, and data is the rotated region
    """
    assert isinstance(rot90,int)
    rot90 = rot90 % 4
    if rot90 == 0:
        return frame,regions
    rotregions = []
    for region in regions:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        data = region[4]
        x,y,w,h = rotate_region(x,y,w,h,frame.shape[1],frame.shape[0],rot90)
        rotdata = np.ascontiguousarray(np.rot90(data,rot90))
        rotregions.append([x,y,w,h,rotdata])
    rotframe = np.ascontiguousarray(np.rot90(frame,rot90))
    return rotframe,rotregions

def get_range_noutfiles(range):
    """
    get_range_noutfiles(range)
    Get the number of output files for a range
    range: None, [start,end], or [[start0,end0],[start1,end1],...]
    Returns:
    -1 if range is None
    0 if range is [start,end]
    len(range) if range is a list of ranges
    """
    if range is None:
        return -1
    if isinstance(range[0],int) and len(range) == 2:
        return 0
    return len(range)

def cropufmf(inmoviefile,croprows=None,cropcols=None,cropframes=None,outmoviefiles=[],outdir=None,rot90=0):
    """
    cropufmf(inmoviefile,croprows=None,cropcols=None,cropframes=None,outmoviefiles=[],outdir=None,rot90=0)
    Crop and rotate a UFMF movie
    inmoviefile: input movie file
    croprows: None, [start,end], or [[start0,end0],[start1,end1],...] to crop rows. Default: None
    cropcols: None, [start,end], or [[start0,end0],[start1,end1],...] to crop columns. Default: None
    cropframes: None, [start,end], or [[start0,end0],[start1,end1],...] to crop frames. Default: None
    outmoviefiles: None or list of output movie files. If None or not enough, names will be autogenerated 
    based on inmoviefile and cropping parameters: 
    <outdir>/<inmoviefile>_crop_row<start0>to<end0>_col<start0>to<end0>_frame<start0>to<end0>_rot<rot90*90>.ufmf
    outdir: None or default output directory
    rot90: number of 90 degree counter-clockwise rotations
    Returns:
    list of output movie files
    """

    rot90 = rot90 % 4

    # open input movie   
    inmovieobj = Movie(inmoviefile)
    inmovieobj.open()
    nframes = inmovieobj.get_n_frames()
    width = inmovieobj.get_width()
    height = inmovieobj.get_height()
    inufmfobj = inmovieobj.h_mov._ufmf
    timestamps = inmovieobj.get_some_timestamps()
    if outmoviefiles is None:
        outmoviefiles = []

    if (outdir is not None) and (not os.path.exists(outdir)):
        os.makedirs(outdir)

    # how many output files
    ncroprows = get_range_noutfiles(croprows)
    ncropcols = get_range_noutfiles(cropcols)
    ncropframes = get_range_noutfiles(cropframes)
    noutfiles = max([ncroprows,ncropcols,ncropframes])
    if noutfiles == -1:
        # no cropping specified, hopefully rot specified
        pass
    else:
        if noutfiles == 0:
            noutfiles = 1
        if ncroprows == 0:
            croprows = [croprows]*noutfiles
        elif ncroprows > 0:
            assert ncroprows == noutfiles, 'nout is not consistent for croprows'
        if ncropcols == 0:
            cropcols = [cropcols]*noutfiles
        elif ncropcols > 0:
            assert ncropcols == noutfiles, 'nout is not consistent for cropcols'
        if ncropframes == 0:
            cropframes = [cropframes]*noutfiles
        elif ncropframes > 0:
            assert ncropframes == noutfiles, 'nout is not consistent for cropframes'
    
    # set default values
    if croprows is None:
        croprows = [[0,-1]]*noutfiles
    if cropcols is None:
        cropcols = [[0,-1]]*noutfiles
    if cropframes is None:
        cropframes = [[0,-1]]*noutfiles

    for i in range(noutfiles):
        if croprows[i][1] < 0:
            croprows[i][1] = height+croprows[i][1]
        if cropcols[i][1] < 0:
            cropcols[i][1] = width+cropcols[i][1]
        if cropframes[i][1] < 0:
            cropframes[i][1] = nframes+cropframes[i][1]
            
        assert croprows[i][0] >= 0 and croprows[i][1] < height and croprows[i][0] <= croprows[i][1], f'Invalid croprows {croprows[i]}'
        assert cropcols[i][0] >= 0 and cropcols[i][1] < width and cropcols[i][0] <= cropcols[i][1], f'Invalid cropcols {cropcols[i]}'
        assert cropframes[i][0] >= 0 and cropframes[i][1] < nframes and cropframes[i][0] <= cropframes[i][1], f'Invalid cropframes {cropframes[i]}'        

    # default output directory is the same as the input directory
    if outdir is None:
        outdir = os.path.dirname(inmoviefile)

    # default output file names
    # split inmoviefile
    _,fn = os.path.split(inmoviefile)
    fnbase,ext = os.path.splitext(fn)
        
    # set output file names
    outmoviefiles = outmoviefiles + [None,]*(noutfiles-len(outmoviefiles))
    for i in range(noutfiles):
        if outmoviefiles[i] is not None:
            continue
        docroprows = croprows[i][0] > 0 or croprows[i][1] < height-1
        docropcols = cropcols[i][0] > 0 or cropcols[i][1] < width-1
        docropframes = cropframes[i][0] > 0 or cropframes[i][1] < nframes-1
        dorotate = rot90 != 0
        outfnbase = fnbase + '_crop'
        if docroprows:
            outfnbase += f'_row{croprows[i][0]}to{croprows[i][1]}'
        if docropcols:
            outfnbase += f'_col{cropcols[i][0]}to{cropcols[i][1]}'
        if docropframes:
            outfnbase += f'_frame{cropframes[i][0]}to{cropframes[i][1]}'        
        if dorotate:
            outfnbase += f'_rot{rot90*90}'
        outmoviefile = os.path.join(outdir,outfnbase+ext)
        outmoviefiles[i] = outmoviefile

    # initialize output movie objects
    outmovieobjs = []
    for i in range(noutfiles):
        outmoviefile = outmoviefiles[i]
        print(f'Creating {outmoviefile}')
        if rot90 % 2 == 0:
            outwidth = cropcols[i][1]-cropcols[i][0]+1
            outheight = croprows[i][1]-croprows[i][0]+1
        else:
            outwidth = croprows[i][1]-croprows[i][0]+1
            outheight = cropcols[i][1]-cropcols[i][0]+1        
        outmovieobj = UfmfSaver(outmoviefile, version=ufmf_ver, coding=inufmfobj.get_coding(),
                                max_width=outwidth,max_height=outheight)
        outmovieobjs.append(outmovieobj)

    # add key frames
    inufmfindex = inufmfobj.get_index()
    keyframetype = 'mean'
    nkeyframes = len(inufmfindex['keyframe'][keyframetype]['loc'])

    # which keyframes to add
    keyframe_timestamps = inufmfindex['keyframe'][keyframetype]['timestamp']
    t0s = []
    t1s = []
    for i in range(noutfiles):
        t1s.append(timestamps[cropframes[i][1]])
        t0s.append(max([t for t in keyframe_timestamps if t <= timestamps[cropframes[i][0]]]))

    for f in range(nkeyframes):
        keyframe,timestamp = inufmfobj._get_keyframe_N(keyframetype,f)
        for i in range(noutfiles):
            if timestamp < t0s[i] or timestamp > t1s[i]:
                continue
            keyframe_crop = keyframe[croprows[i][0]:croprows[i][1]+1,cropcols[i][0]:cropcols[i][1]+1]
            if rot90 != 0:
                keyframe_crop = np.ascontiguousarray(np.rot90(keyframe_crop,rot90))
            outmovieobjs[i].add_keyframe(keyframetype,keyframe_crop,timestamp)

    # add frames
    minframe = min([c[0] for c in cropframes])
    maxframe = max([c[1] for c in cropframes])
    for f in tqdm.trange(minframe,maxframe+1):
        frame,timestamp,more = inmovieobj.h_mov.get_frame(f,_return_more=True)
        for i in range(noutfiles):
            if f < cropframes[i][0] or f > cropframes[i][1]:
                continue
            cfr,cropregions = cropframe(frame,more,croprows[i],cropcols[i])
            if rot90 != 0:
                cfr,cropregions = rotateframe(cfr,cropregions,rot90)
            _ = outmovieobjs[i].add_frame(cfr,timestamp,cropregions,iscentered=False)


    # close
    for i in range(noutfiles):
        print(f'Closing {outmoviefiles[i]}')
        outmovieobjs[i].close()
        
    inmovieobj.close()
    
    return {'outmoviefiles': outmoviefiles,
            'croprows': croprows,
            'cropcols': cropcols,
            'cropframes': cropframes,
            'rot90': rot90,
            'outdir': outdir}
    
def validate_args(args):
    """
    validate_args(args)
    Check that command line arguments are formatted properly. There will also be some additional checks
    in the main function.
    Checks that:
    - inmoviefile exists
    - croprows, cropcols, cropframes are a list of length 2 or a lists of lists of length 2
    - croprows, cropcols, cropframes values are integers
    Converts tuples to lists
    """
    if not os.path.exists(args.inmoviefile):
        raise ValueError(f"Input file not found: {args.inmoviefile}")

    def validate_crop_ranges(ranges, name):
        if ranges is None:
            return ranges
        if isinstance(ranges, tuple):
            ranges = list(ranges)
        if not isinstance(ranges, list):
            raise ValueError(f"{name} must be a list/tuple of tuples")
        # single range will be universally applied
        if isinstance(ranges[0], int) and len(ranges) == 2:
            return ranges
        for r in ranges:
            if isinstance(r, tuple):
                r = list(r)
            if not isinstance(r, list) or len(r) != 2:
                raise ValueError(f"Each {name} range must be a list of 2 integers")
            for x in r:
                if not isinstance(x, (int, np.integer)):
                    raise ValueError(f"{name} range values must be integers")
        return ranges

    # Convert and validate
    args.croprows = validate_crop_ranges(args.croprows, "croprows")
    args.cropcols = validate_crop_ranges(args.cropcols, "cropcols")
    args.cropframes = validate_crop_ranges(args.cropframes, "cropframes")
        
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description='Crop UFMF movie file',
        epilog='Example: python crop_ufmf.py /path/to/inmoviefile.ufmf --croprows "[[0,100],[101,-1]]" --cropcols "[(0,200),(201,-1)]" --cropframes "[[200,300],[301,-1]]" --outmoviefiles /path/to/outfile1.ufmf /path/to/outfile2.ufmf --rot90 1'
    )
    parser.add_argument('inmoviefile', help='Input movie file')
    parser.add_argument('--croprows', type=ast.literal_eval, help='Row crop ranges [[start0,end0], [start1,end1],...]')
    parser.add_argument('--cropcols', type=ast.literal_eval, help='Column crop ranges [[start0,end0], [start1,end1],...]')
    parser.add_argument('--cropframes', type=ast.literal_eval, help='Frame crop ranges [[start0,end0], [start1,end1],...]')
    parser.add_argument('--outmoviefiles', nargs='+', help='Output movie filenames')
    parser.add_argument('--outdir', default=None, help='Output directory')
    parser.add_argument('--rot90', default=0, type=int, help='Number of counter-clockwise 90 degree rotations')

    args = parser.parse_args()
    
    # print all args
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    
    outmoviefiles = cropufmf(args.inmoviefile,croprows=args.croprows,cropcols=args.cropcols,cropframes=args.cropframes,
                            outmoviefiles=args.outmoviefiles,outdir=args.outdir,rot90=args.rot90)
    
    print('Finished')