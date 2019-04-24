% header = mmf_read_header(filename)
% TODO: handle multichannel data, non 8-bit data

% mmf format documentation:
% Set of Image Stacks representing a movie. Beginning of file is a header, with this format:
% 10240 byte zero padded header beginning with a textual description of the file, followed by \0 then the following fields (all ints, except idcode)
% 4 byte unsigned long idcode = a3d2d45d, header size in bytes, key frame interval, threshold below background, threshold above background
% Header is followed by a set of common background image stacks, with the following format:
% Stack of common background images, beginning with this header:
% 512 byte zero-padded header, with the following fields (all 4 byte ints, except idcode):
% 4 byte unsigned long idcode = bb67ca20, header size in bytes, total size of stack on disk, nframes: number of images in stack
% Then the background image, as an IplImage, starting with the 112 byte image header, followed by the image data
% Then nframes background removed images containing only differences from the background, in this format:
% BackgroundRemovedImage: header is a1024 byte zero padded header with the following data fields (all 4 byte ints, except id code)
% 4 byte unsigned long idcode = f80921af, headersize (number of bytes in header), depth (IplImage depth), nChannels (IplImage number of channels), numims (number of 
% image blocks that differ from background) then metadata:
% Name-Value MetaData: idcode (unsigned long) = c15ac674, int number of key-value pairs stored, then each pair
% in the format \0-terminated string of chars then 8 byte double value
% header is followed by numims image blocks of the following form:
% (16 bytes) CvRect [x y w h] describing location of image data, then interlaced row ordered image data

function header = mmf_read_header(filename,varargin)

% debug
% filename = '/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/data/larvae_bruno/20120211_161800/20120211_161800@FCF_w1118_1500005@UAS_TNT_2_0003@t7@n#n#n#n@30@.mmf'

INDEXEXT = '.index.mat';
BGMARKER = hex2dec('bb67ca20');
IMAGEMARKER = hex2dec('f80921af');
DEBUG = false;
MAXNMEANSCACHED = 5;

[forcebuildindex,DEBUG] = myparse(varargin,'forcebuildindex',false,'debug',DEBUG);

% name of file with frame to file index
indexfilename = [filename,INDEXEXT];
dobuildindex = forcebuildindex || ~exist(indexfilename,'file');

header = struct;
header.filename = filename;

% open file
fid = fopen(filename,'rb');
if fid < 0,
  error('Error opening file %s',filename);
end
header.fid = fid;

if ~dobuildindex,
  header = load(indexfilename);
  header.fid = fid;
else
  
% Set of Image Stacks representing a movie. Beginning of file is a header, with this format:
% 10240 byte zero padded header beginning with a textual description of the file, followed by \0 then the following fields (all ints, except idcode)

% read in the textual description of the file
while true,
  fgets(fid);
  t = fread(fid,1,'char');
  if t == 0,
    break;
  end
  fseek(fid,-1,'cof');
end

% read idcode
header_idcode = dec2hex(fread(fid,1,'ulong')); %#ok<NASGU>

% read header size
headersize = fread(fid,1,'int');
header.headersize = headersize;

% read keyframe interval
keyframeinterval = fread(fid,1,'int');
header.keyframeinterval = keyframeinterval;

% read threshold below background
thresholdbelowbackground = fread(fid,1,'int'); %#ok<NASGU>

% read threshold below background
thresholdabovebackground = fread(fid,1,'int'); %#ok<NASGU>

% seek to the end of the header
fseek(fid,headersize,'bof');

header.stacksizes = [];
header.stacknframes = [];
header.stackstarts = [];
header.mean2file = [];
header.frame2file = [];
header.frame2mean = [];

isfirst = true;
stacki = 0;

while true,
  
  % Stack of common background images, beginning with this header:
  % 512 byte zero-padded header, with the following fields (all 4 byte ints, except idcode):
  % 4 byte unsigned long idcode = bb67ca20, header size in bytes, total size of stack on disk, nframes: number of images in stack

  stackstart = ftell(fid);
  
  bg_idcode = fread(fid,1,'ulong'); 
  if isempty(bg_idcode) || bg_idcode ~= BGMARKER,
    break;
  end
  
  stacki = stacki + 1;

  header.stackstarts(end+1) = stackstart;
  stackheadersize = fread(fid,1,'int');

  stacksize = fread(fid,1,'int');
  header.stacksizes(end+1) = stacksize;
  stacknframes = fread(fid,1,'int');
  header.stacknframes(end+1) = stacknframes;

  % seek to the end of this header
  fseek(fid,stackstart+stackheadersize,'bof');

  % Then the background image, as an IplImage, starting with the 112 byte image header, followed by the image data
  backgroundimageheadersize = fread(fid,1,'int');
  if isfirst,
    bkgdim_id = fread(fid,1,'int');
    bkgdim_nchannels = fread(fid,1,'int');
    if bkgdim_nchannels ~= 1,
      error('Cannot read multichannel data yet: TODO');
    end
    bkgdim_alphachannel = fread(fid,1,'int');
    bkgdim_depth = fread(fid,1,'int');
    if bkgdim_depth ~= 8,
      error('Cannot read non-8-bit depth images yet: TODO');
    end
    bkgdim_colormodel = fread(fid,4,'char=>char')';
    bkgdim_channelseq = fread(fid,4,'char=>char')';
    % 0 - interleaved color channels, 1 - separate color channels
    bkgdim_dataorder = fread(fid,1,'int');
    % 0 - top-left origin, 1 - bottom-left origin (Windows bitmaps style)
    bkgdim_origin = fread(fid,1,'int');
    % Alignment of image rows (4 or 8). OpenCV ignores it and uses widthStep instead
    bkgdim_align = fread(fid,1,'int');
    bkgdim_width = fread(fid,1,'int');
    bkgdim_height = fread(fid,1,'int');
    % image ROI. if NULL, the whole image is selected
    bkgdim_roi = fread(fid,1,'uint');
    % must be NULL
    bkgdim_maskroi = fread(fid,1,'uint');
    % must be NULL
    bkgdim_imageid = fread(fid,1,'uint');
    % must be NULL
    bkgdim_tileinfo = fread(fid,1,'uint');
    % image data size in bytes
    % (==image->height*image->widthStep
    % in case of interleaved data)
    bkgdim_imagesize = fread(fid,1,'int');
    bkgdim_imagedata = fread(fid,1,'uint');
    % size of aligned image row in bytes
    bkgdim_widthstep = fread(fid,1,'int');
    % ignored
    bkgdim_bordermode = fread(fid,4,'int');
    % ignored
    bkgdim_borderconst = fread(fid,4,'int');
    % pointer to something
    bkgdim_imagedataorigin = fread(fid,1,'uint');
    
    header.width = bkgdim_height;
    header.height = bkgdim_width;
    header.bkgdim_imagesize = bkgdim_imagesize;
    header.bkgdim_widthstep = bkgdim_widthstep;
    isfirst = false;
  else
    fseek(fid,header.stackstarts(end)+stackheadersize+backgroundimageheadersize,'bof');
  end

  header.mean2file(end+1) = ftell(fid);
  
  % followed by the image data
  if DEBUG,
    bkgdim_data = fread(fid,bkgdim_imagesize,'uint8=>uint8');
    bkgdim_data = reshape(bkgdim_data,bkgdim_widthstep,bkgdim_height);
    bkgdim_data = bkgdim_data(1:bkgdim_width,:);
  else
    fseek(fid,bkgdim_imagesize,'cof');
  end
  
  % Then nframes background removed images containing only differences from the background, in this format:
  % BackgroundRemovedImage: header is a 1024 byte zero padded header with the following data fields (all 4 byte ints, except id code)
  % 4 byte unsigned long idcode = f80921af, headersize (number of bytes in header), depth (IplImage depth), nChannels (IplImage number of channels), numims (number of
  % image blocks that differ from background) then metadata:
  % Name-Value MetaData: idcode (unsigned long) = c15ac674, int number of key-value pairs stored, then each pair
  % in the format \0-terminated string of chars then 8 byte double value
  for i = 1:stacknframes,
    header.frame2file(end+1) = ftell(fid);
    header.frame2mean(end+1) = stacki;
    im_idcode = fread(fid,1,'ulong');
    if im_idcode ~= IMAGEMARKER,
      break;
    end
    im_headersize = fread(fid,1,'int');
    %im_depth = fread(fid,1,'int');
    %im_nchannels = fread(fid,1,'int');
    fseek(fid,8,'cof');
    im_numblocks = fread(fid,1,'int');
    %im_metadata_idcode = fread(fid,1,'ulong');
    %im_metadata_nkeyvals = fread(fid,1,'int');
    %im_metadata_keys = cell(1,im_metadata_nkeyvals);
    %im_metadata_vals = cell(1,im_metadata_nkeyvals);
    %for i = 1:im_metadata_nkeyvals,
    %fn = '';
    %while true,
    %  c = fread(fid,1,'*char');
    %  if c == 0,
    %    break;
    %  end
    %  fn(end+1) = c; %#ok<AGROW>
    %end
    %val = fread(fid,1,'double');
    %im_metadata_keys{i} = fn;
    %im_metadata_vals{i} = val;
    %end
    
    % seek to the end of the header
    fseek(fid,header.frame2file(end)+im_headersize,'bof');
    if DEBUG,
      im_data = bkgdim_data;
    end
    for j = 1:im_numblocks,
      block_rect = fread(fid,4,'int');
      % then interlaced row ordered image data
      if DEBUG,
        block_data = fread(fid,block_rect(3)*block_rect(4),'*uint8');
        block_data = reshape(block_data,[block_rect(3),block_rect(4)]);
        im_data(block_rect(1)+1:block_rect(1)+block_rect(3),block_rect(2)+1:block_rect(2)+block_rect(4)) = block_data;
      else
        fseek(fid,block_rect(3)*block_rect(4),'cof');
      end
    end
    
    if DEBUG,
      imagesc(im_data);
      axis image;
      drawnow;
    end
    
  end
  
end

header.nframes = numel(header.frame2file);
header.nmeans = numel(header.mean2file);
header.ncolors = 1;
header.nr = header.height;
header.nc = header.width;

end

try
  save(indexfilename,'-struct','header');
catch ME
  warning('Could not save index file to %s:\n%s',indexfilename,getReport(ME));
end

% cache some means
% allocate cache
nmeanscached = min(MAXNMEANSCACHED,header.nmeans);
header.cachedmeans = zeros([header.nr,header.nc,nmeanscached],'uint8');
header.cachedmeans_idx = zeros(1,nmeanscached);
header.cachedmeans_accesstime = -inf(1,nmeanscached);
% read in the means; this automatically stores them in the cache
for i = 1:nmeanscached,
  [~,header] = mmf_read_mean(header,'meani',i);
end

