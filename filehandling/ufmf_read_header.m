% header = ufmf_read_header(filename)
function header = ufmf_read_header(filename)

MAXNMEANSCACHED = 5;

header.filename = filename;

% open the file for reading, binary, little-endian
fp = fopen( filename, 'rb' , 'ieee-le');
if ~exist(filename,'file'),
  error('File %s does not exist',filename);
end
if fp < 0,
  error('File %s exists, but could not open it in rb mode',filename);
end
header.fid = fp;

% ufmf: 4
s = fread(fp,[1,4],'*char');
if ~strcmp(s,'ufmf'),
  error('Invalid UFMF file: first four bytes must be ''ufmf''.');
end

% version: 4
header.version = fread(fp,1,'uint');
if header.version < 2,
  error('Only UFMF versions 2-4 are supported.');
end

% index location: 8
header.indexloc = fread(fp,1,'uint64');

% this is somewhat backwards for faster reading
% max_height: 2, max_width: 2
sz = fread(fp,2,'ushort');
header.max_height = sz(1); header.max_width = sz(2);

% whether it is fixed size patches: 1
if header.version >= 4,
  header.is_fixed_size = fread(fp,1,'uchar');
else
  header.is_fixed_size = false;
end

% coding: length(coding)
l = fread(fp,1,'uchar');
header.coding = fread(fp,[1,l],'*char');
switch lower(header.coding),
  case 'mono8',
    header.ncolors = 1;
    header.bytes_per_pixel = 1;
  case 'rgb24',
    header.ncolors = 3;
    header.bytes_per_pixel = 3;
end
header.dataclass = 'uint8'; 

% seek to the start of the index
fseek(fp,header.indexloc,'bof');

% read in the index
index = read_dict(fp);

% frame number to loc
header.frame2file = index.frame.loc;
header.nframes = length(header.frame2file);
header.timestamps = index.frame.timestamp;

% mean number to loc
header.mean2file = index.keyframe.mean.loc;
header.nmeans = length(header.mean2file);
header.mean_timestamps = index.keyframe.mean.timestamp;

% frame number to mean loc
header.frame2mean = zeros(header.nframes,1);
header.frame2mean(:) = header.nmeans;
for i = 1:header.nmeans-1,
  idx = header.timestamps >= header.mean_timestamps(i) & ...
    header.timestamps < header.mean_timestamps(i+1);
  header.frame2mean(idx) = i;
end
header.frame2meanloc = header.mean2file(header.frame2mean);

% get the frame size: read in the first mean image
[mean1,header] = ufmf_read_mean(header,'meani',1);
[header.nr,header.nc,~] = size(mean1);
header.meandataclass = class(mean1);

% cache some means
% allocate cache
nmeanscached = min(MAXNMEANSCACHED,header.nmeans);
header.cachedmeans = zeros([header.ncolors,header.nr,header.nc,nmeanscached],header.dataclass);
header.cachedmeans_idx = zeros(1,nmeanscached);
header.cachedmeans_accesstime = -inf(1,nmeanscached);
% read in the means; this automatically stores them in the cache
for i = 1:nmeanscached,
  [~,header] = ufmf_read_mean(header,'meani',i,'dopermute',false);
end

function index = read_dict(fp)

DICT_START_CHAR = 'd';
ARRAY_START_CHAR = 'a';

% read in a 'd': 1
chunktype = fread(fp,1,'*char');
if chunktype ~= DICT_START_CHAR,
  error('Error reading index: dictionary does not start with ''%s''.',DICT_START_CHAR);
end

% read in the number of fields: 1
nkeys = fread(fp,1,'uchar');

for j = 1:nkeys,
  
  % read the length of the key name: 2
  l = fread(fp,1,'ushort');
  % read the key name: l
  key = fread(fp,[1,l],'*char');
  % read the next letter to tell if it is an array or another dictionary
  chunktype = fread(fp,1,'*char');
  if chunktype == DICT_START_CHAR,
    % if it's a 'd', then step back one char and read in the dictionary
    % recursively
    fseek(fp,-1,'cof');
    index.(key) = read_dict(fp);
  elseif chunktype == ARRAY_START_CHAR,
    % array
    
    % read in the data type
    dtypechar = fread(fp,1,'*char');
    [matlabclass,bytes_per_element] = dtypechar2matlabclass(dtypechar);
    
    % read in number of bytes
    l = fread(fp,1,'ulong');
    n = l / bytes_per_element;
    if n ~= round(n),
      error('Length in bytes %d is not divisible by bytes per element %d',l,bytes_per_element);
    end
    
    % read in the index array
    [index.(key),ntrue] = fread(fp,n,['*',matlabclass]);
    if ntrue ~= n,
      warning('Could only read %d/%d bytes for array %s of index',n,ntrue,key);
    end
    
  else
    
    error('Error reading dictionary %s. Expected either ''%s'' or ''%s''.',...
      key,DICT_START_CHAR,ARRAY_START_CHAR);
    
  end

end
