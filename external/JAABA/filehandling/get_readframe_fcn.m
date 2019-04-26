% [readframe,nframes,fid,headerinfo] = get_readframe_fcn(filename)
% get_readframe_fcn inputs the name of a video, opens this video, and creates a
% function that can be used for random access to frames in the video. it can
% input files of type fmf, sbfmf, and avi. if videoio is installed on your
% machine, then get_readframe_fcn will use videoio to read all types of avi
% supported by videoio. otherwise, it uses VideoReader to open any type of avi
% supported by VideoReader. Run "help VideoReader" to determine what types of avi
% files are readable through VideoReader on your OS. 
%
% Input:
% filename is the name of the video to read. The type of video is determined
% based on the extension, so be sure the extension of your movie is correct. 
%
% Outputs:
% readframe is the handle to a function which inputs a frame number, reads this
% frame from filename, and outputs the corresponding frame. 
% nframes is the number of frames in the video. For avis, this may be
% approximate, as it is computed from the duration and framerate. 
% fid is the file identifier for filename. get_readframe_fcn will open the file
% filename. you must close fid yourself when you are done using the returned
% readframe. 
% headerinfo is a struct with fields depending on the type of movie

function [readframe,nframes,fid,headerinfo] = get_readframe_fcn(filename,varargin)

i = find(strcmpi(varargin(1:2:end),'preload'));
if ~isempty(i),
  preload = varargin{i+1};
  varargin(i:i+1) = [];
else
  preload = false;
end
didpreload = false;

% allow videoio library to be used if it is installed and on the path
%CTRAX_ISVIDEOIO = exist('videoReader','file');
CTRAX_ISVIDEOIO = false;

if iscell(filename),
  
  readframes = cell(size(filename));
  nframes = inf;
  fid = nan(size(filename));
  headerinfo = cell(size(filename));
  for i = 1:numel(filename),
    [readframes{i},nframescurr,fid(i),headerinfo{i}] = get_readframe_fcn(filename{i},varargin{:});
    nframes = min(nframes,nframescurr);
  end
  
  readframe = @(f) multi_read_frame(f,readframes);
  return;
  
end

[~,ext] = splitext(filename);

if ispc && ~exist(filename,'file') && ~strcmpi(ext,'.seq'),  
  [actualfilename,didfind] = GetPCShortcutFileActualPath(filename);
  if didfind,
    filename = actualfilename;
  end
end

if strcmpi(ext,'.fmf'),
  [header_size,version,nr,nc,bytes_per_chunk,nframes,data_format] = fmf_read_header(filename);
  fid = fopen(filename);
  readframe = @(f) fmfreadframe(fid,header_size+(f-1)*bytes_per_chunk,nr,nc,bytes_per_chunk,data_format);
  headerinfo = struct('header_size',header_size,'version',version,'nr',nr,'nc',nc,...
    'bytes_per_chunk',bytes_per_chunk,'nframes',nframes,'data_format',data_format,'type','fmf');
elseif strcmpi(ext,'.sbfmf'),
  [nr,nc,nframes,bgcenter,bgstd,frame2file] = sbfmf_read_header(filename);
  fid = fopen(filename);
  readframe = @(f) sbfmfreadframe(f,fid,frame2file,bgcenter);
  headerinfo = struct('nr',nr,'nc',nc,'nframes',nframes,'bgcenter',bgcenter,...
    'bgstd',bgstd,'frame2file',frame2file,'type','sbfmf');
elseif strcmpi(ext,'.ufmf'),
  headerinfo = ufmf_read_header(filename);
  readframe = ufmf_get_readframe_fcn(headerinfo,varargin{:});
  nframes = headerinfo.nframes;
  fid = headerinfo.fid;
elseif strcmpi(ext,'.mmf'),
  headerinfo = mmf_read_header(filename);
  readframe = mmf_get_readframe_fcn(headerinfo,varargin{:});
  nframes = headerinfo.nframes;
  fid = headerinfo.fid;
elseif strcmpi(ext,'.tif') || strcmp(ext,'.tiff'),
  info = imfinfo(filename);
  isimseq = false;
  if numel(info) == 1,
    filespec = regexprep(filename,['_\d+\.',ext,'$'],['_*.',ext]);
    imfiles = mydir(filespec);
    if numel(imfiles) > 1,
      imfiles = sort(imfiles);
      im = imread(imfiles{1});
      headerinfo = struct('nr',size(im,1),'nc',size(im,2),'ncolors',size(im,3),'nframes',numel(imfiles),...
        'type','imseq','imfiles',{imfiles});
      readframe = @(f) imseq_read_frame(f,imfiles);
      nframes = headerinfo.nframes;
      fid = -1;
      isimseq = true;
    end
  end
  if ~isimseq,
    headerinfo = struct('nr',info(1).Height,'nc',info(1).Width,'nframes',numel(info),'type','tif',...
      'bitdepth',info(1).BitDepth);
    readframe = @(f) deal(imread(filename,f),f);
    nframes = headerinfo.nframes;
    fid = -1;
  end
elseif strcmpi(ext,'.png'),
  info = imfinfo(filename);
  isimseq = false;
  filespec = regexprep(filename,'_\d+\.png$','_*.png');
  imfiles = mydir(filespec);
  if numel(imfiles) > 1,
    imfiles = sort(imfiles);
    im = imread(imfiles{1});
    headerinfo = struct('nr',size(im,1),'nc',size(im,2),'ncolors',size(im,3),'nframes',numel(imfiles),...
      'type','imseq','imfiles',{imfiles});
    readframe = @(f) imseq_read_frame(f,imfiles);
    nframes = headerinfo.nframes;
    fid = -1;
    isimseq = true;
  end
elseif strcmpi(ext,'.klb'),
  tfKLBLib = exist('readKLBslice','file')>0;
  if ~tfKLBLib
    error('get_readframe_fcn:klb','Cannot find readKLBslice function. Make sure KLB matlab library is installed.');
  end
  
  dim = myparse(varargin,'dim',3);
  headerinfo = readKLBheader(filename);
  readframe = klb_get_readframe_fcn(filename,varargin{:});
  nframes = headerinfo.xyzct(dim); % AL 201507: KLB doc a little unclear here but this seems right
  fid = 0;
elseif strcmpi(ext,'.mat'),
  videofiletype = load(filename,'videofiletype');
  switch videofiletype,
    
    case 'SingleLarvaTracker',
      videodata = load(filename);
      readframe = @(f) ReadSingleLarvaTrackerFrame(f,videodata.firstframeim,videodata.imraw,videodata.finalbbox,videodata.fps,varargin{:});
      nframes = numel(imraw);
      fid = 0;
      [nr,nc,~] = size(firstframeim);
      headerinfo = struct('nr',nr,'nc',nc,'nframes',nframes,'bgcenter',firstframeim,...
        'type','SingleLarvaTracker');
    otherwise
      error('Do not know how to parse mat file of type %s',videofiletype);
  end

elseif strcmpi(ext,'.seq'),
  [indexfilename,seqtype] = myparse(varargin,'indexfilename',0,'seqtype',0);
  
  if ischar(indexfilename),
    seqtype = 'shay';
  end
  
  % get file names
  if ispc && ~exist(filename,'file'),
      
    [actualfilename,didfind] = GetPCShortcutFileActualPath(filename);
    if ~didfind,
      error('Could not find movie file %s',filename);
    end
    filename0 = filename;
    % use actualfilename instead
    filename = actualfilename;
      
    % try the index file using the original filename
    if ~ischar(indexfilename),
      indexfilename = regexprep(filename0,'seq$','mat');
      [indexfilename,didfind] = GetPCShortcutFileActualPath(indexfilename);
      if ~didfind && ~strcmp(filename0,filename),
        % try to find the index file using the source filename
        indexfilename = regexprep(filename,'seq$','mat');
        [indexfilename,didfind] = GetPCShortcutFileActualPath(indexfilename);
        % if didn't find index file, then maybe this is a piotr seq file
        if ~didfind,
          if ischar(seqtype) && strcmpi(seqtype,'shay'),
            error('Could not find index file for %s',filename);
          else
            seqtype = 'piotr';
          end
        end
      end
    end
    
  else
    
    % set seqtype to piotr if indexfilename doesn't exist
    if ~ischar(seqtype) && ~ischar(indexfilename),
      indexfilename = regexprep(filename,'seq$','mat');
      if ~exist(indexfilename,'file'),
        seqtype = 'piotr';
      end
    end
    
  end
    
  % get actual filename for shortcuts
  if strcmpi(seqtype,'piotr'),
    warning('Closing file currently not implemented for Piotr''s seq files...');
    headerinfo = seqIo(filename,'getinfo');
    headerinfo.nr = headerinfo.height;
    headerinfo.nc = headerinfo.width;
    headerinfo.nframes = headerinfo.numFrames;
    headerinfo.type = 'seq_piotr';
    fid = 0;
    nframes = headerinfo.numFrames;
    sr = seqIo(filename,'r');
    readframe = @(f) seq_read_frame_piotr(f,sr);
  else
    
    if ischar(indexfilename),
      headerinfo = r_readseqinfo(filename,indexfilename);
    else
      headerinfo = r_readseqinfo(filename);
    end
    if numel(headerinfo.m_aiSeekPos) < headerinfo.m_iNumFrames,
      headerinfo.m_iNumFrames = numel(headerinfo.m_aiSeekPos);
    end
    headerinfo.nr = headerinfo.m_iHeight;
    headerinfo.nc = headerinfo.m_iWidth;
    headerinfo.nframes = headerinfo.m_iNumFrames;
    headerinfo.type = 'seq';
    fid = 0;
    nframes = headerinfo.m_iNumFrames;
    readframe = @(f) read_seq_frame(headerinfo,f);
  end
  
elseif strcmpi(ext,'.h5'),
  
  [dims,datasetname,fixdims,fixidx] = myparse(varargin,'slicedims',3,'datasetname',[],'fixdims',[],'fixidx',[]);
  if isempty(datasetname),
    headerinfo = h5info(filename);
    datasetname = headerinfo.Name;
    headerinfo = headerinfo.Datasets;
  else
    headerinfo = h5info(filename,datasetname);
  end
  headerinfo.datasetname = datasetname;
  headerinfo.dims = dims;
  headerinfo.fixdims = fixdims;
  headerinfo.fixidx = fixidx;
  headerinfo.sz = headerinfo.Dataspace.Size;
  headerinfo.ndims = numel(headerinfo.sz);
  headerinfo.nframes = headerinfo.sz(headerinfo.dims);

  readframe = h5_get_readframe_fcn(filename,headerinfo);
  nframes = headerinfo.nframes;
  fid = 0;
  
else
  fid = 0;
  
  isindexedmjpg = false;
  if strcmpi(ext,'.mjpg'),
    [indexfilename] = myparse(varargin,'indexfilename',[]);
    if isempty(indexfilename),
      % see if this looks like an indexed mjpg file
      [p,n] = fileparts(filename);
      indexfilename = fullfile(p,[n,'.txt']);
      if ~exist(indexfilename,'file') && ispc,
        [actualindexfilename,didfind] = GetPCShortcutFileActualPath(indexfilename);
        if didfind,
          indexfilename = actualindexfilename;
        end
      end
      if exist(indexfilename,'file'),
        try
          ReadIndexedMJPGHeader(filename,indexfilename);
          isindexedmjpg = true;
        catch
          indexfilename = 0;
        end
      else
        indexfilename = 0;
      end
    elseif ischar(indexfilename),
      isindexedmjpg = true;
    end
  end
  
  if isindexedmjpg,
    % get file names
    if ispc && ~exist(filename,'file'),
      [actualfilename,didfind] = GetPCShortcutFileActualPath(filename);
      if ~didfind,
        error('Could not find movie file %s',filename);
      end
      filename0 = filename;
      % use actualfilename instead
      filename = actualfilename;

      [actualindexfilename,didfind] = GetPCShortcutFileActualPath(indexfilename);
      if ~didfind,
        error('Could not find index file %s',indexfilename);
      end
      indexfilename0 = indexfilename;
      % use actualfilename instead
      indexfilename = actualindexfilename;
    end

    headerinfo = ReadIndexedMJPGHeader(filename,indexfilename);
    headerinfo.fid = 0;
    nframes = headerinfo.nframes;
    readframe = @(f) read_mjpg_frame(headerinfo,f);
  
  else
  
    
    if CTRAX_ISVIDEOIO,
      readerobj = videoReader(filename,'preciseFrames',30,'frameTimeoutMS',5000);
      info = getinfo(readerobj);
      nframes = info.numFrames;
      seek(readerobj,0);
      seek(readerobj,1);
      readframe = @(f) videoioreadframe(readerobj,f);
      headerinfo = info;
      headerinfo.type = 'avi';
    else
      
      % optional argument to not use the read function of VideoReader
      [useRead,neednframes] = myparse(varargin,'useRead',true,'neednframes',true);
      
      try
        readerobj = VideoReader(filename);
        headerinfo = get(readerobj);
        
        headerinfo.type = 'avi';
        headerinfo.nr = headerinfo.Height;
        headerinfo.nc = headerinfo.Width;
        
        if preload,
          
          
          nframes = 0;
          preloaddata = struct;
          preloaddata.ims = {};
          preloaddata.ts = [];
          
          while true,
          
            if ~hasFrame(readerobj),
              break;
            end
            preloaddata.ts(end+1) = get(readerobj,'CurrentTime');
            preloaddata.ims{end+1} = readFrame(readerobj);
            nframes = nframes+1;
            
          end
            
          assert(numel(preloaddata.ims) == nframes);
          
        else
          if neednframes && useRead,
            nframes = get(readerobj,'NumberOfFrames');
          end
          if ~neednframes || ~useRead || isempty(nframes),
            % approximate nframes from duration
            nframes = round(get(readerobj,'Duration')*get(readerobj,'FrameRate'));
          end
          headerinfo.readerobj = readerobj;
          
        end

        headerinfo.nframes = nframes;
        
        if preload,
          
          readframe = @(f) preload_read_frame(preloaddata,f);
          didpreload = true;
          
        else
        
          if useRead,
            readframe = @(f) avi_read_frame(readerobj,headerinfo,f);
          else
            readframe = @(f) avi_read_frame_useReadFrame(readerobj,headerinfo,f);
          end
          
        end
      catch ME_videoreader
        error('Could not open file %s with VideoReader: %s',...
          filename,getReport(ME_videoreader));
      end
    end
  end
end

if preload && ~didpreload,
  
  preloaddata = struct;
  preloaddata.ims = cell(1,headerinfo.nframes);
  preloaddata.ts = nan(1,headerinfo.nframes);
          
  for f = 1:headerinfo.nframes,
    [preloaddata.ims{f},preloaddata.ts(f)] = readframe(f);
  end
  
  readframe = @(f) preload_read_frame(preloaddata,f);
  didpreload = true;
  
end

function varargout = multi_read_frame(f,readframes)

[im,timestamp] = readframes{1}(f);

for i = 2:numel(readframes),
  im = cat(2,im,readframes{i}(f));  
end
varargout{1} = im;
if nargout >= 2,
  varargout{2} = timestamp;
end

function [im,timestamp] = seq_read_frame_piotr(f,sr)
im = [];
timestamp = [];

if ~sr.seek(f-1),
  warning('Could not seek to frame %d',f);
  return;
end

[im,timestamp] = sr.getframe();

function [im,timestamp] = avi_read_frame_useReadFrame(readerobj,headerinfo,f)

timestamp = (f-1)/headerinfo.FrameRate;
oldtimestamp = get(readerobj,'CurrentTime');
if oldtimestamp ~= timestamp,
  set(readerobj,'CurrentTime',timestamp);
end
timestamp = get(readerobj,'CurrentTime');
im = readFrame(readerobj);

function [im,timestamp] = preload_read_frame(preloaddata,f)

try 
  im = preloaddata.ims{f};
  timestamp = preloaddata.ts(f);
catch ME,
  if ~isempty(preloaddata.ims),
    im = preloaddata.ims{end};
    timestamp = preloaddata.ts(end);
  end
  warning('Error reading preloaded frame %d: %s',f,getReport(ME));
end

function [im,timestamp] = avi_read_frame(readerobj,headerinfo,f)

try
  im = read(readerobj,f);
catch ME,
  warning('Error reading the first try: %s',getReport(ME));
  pause(.01);
  im = read(readerobj,f);
end
timestamp = (f-1)/headerinfo.FrameRate;

function [im,f] = imseq_read_frame(f,imfiles)

im = imread(imfiles{f});
