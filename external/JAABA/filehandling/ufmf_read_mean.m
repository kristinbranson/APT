function [im,header] = ufmf_read_mean(header,varargin)

[meani,framei,dopermute] = myparse(varargin,'meani',[],'framei',[],'dopermute',true);
fp = header.fid;

meani = [meani,header.frame2mean(framei)];

if isempty(meani),
  [im,header] = ufmf_read_mean_helper(fp,header);
else
  if length(meani) > 1,
    im = zeros([header.ncolors,header.nr,header.nc,length(meani)],header.meandataclass);
  end
  if isfield(header,'cachedmeans_idx'),
    [idx,cachei] = ismember(meani,header.cachedmeans_idx);
  else
    idx = false(size(meani));
  end
  for i = find(idx),
    im(:,:,:,i) = header.cachedmeans(:,:,:,cachei(i));
    header.cachedmeans_accesstime(i) = now;
  end
  for i = find(~idx),
    fseek(fp,header.mean2file(meani(i)),'bof');
    [im(:,:,:,i),header] = ufmf_read_mean_helper(fp,header);
  end
end
if dopermute,
  im = permute(im,[2,3,1,4]);
end

function [im,header,timestamp] = ufmf_read_mean_helper(fp,header)

% Keyframe file format:
%
% 0 (chunk type)                       uchar
% 4 (length of keyframe type)          uchar
% 'mean' (keyframe type)               char x 4
% timestamp                            double
% number of boxes/points               ushort
% dtype ('f' for float, 'B' for uint8) char
% width                                ushort
% height                               ushort
% timestamp                            double
% background mean                      dtype x width x height x ncolors
% (iterate over colors, 
%  followed by columns in sideways im,
%  followed by rows)

KEYFRAME_CHUNK = 0;
MEAN_KEYFRAME_TYPE = 'mean';

loc = ftell(fp);
meani = find(header.mean2file == loc,1);
if isempty(meani),
  error('Could not find current file location in mean2file index');
end

% chunktype: 1
chunktype = fread(fp,1,'uchar');
if chunktype ~= KEYFRAME_CHUNK,
  error('Expected chunktype = %d at start of keyframe.');
end

% keyframe type
l = fread(fp,1,'uchar');
keyframe_type = fread(fp,[1,l],'*char');
if ~strcmp(keyframe_type,MEAN_KEYFRAME_TYPE),
  error('Expected keyframe type = ''%s'' at start of mean keyframe',MEAN_KEYFRAME_TYPE);
end

% data type
dtypechar = fread(fp,1,'*char');
matlabclass = dtypechar2matlabclass(dtypechar);

% images are sideways: swap width and height
% width, height
sz = double(fread(fp,2,'ushort'));
height = sz(1); width = sz(2);

% timestamp
timestamp = fread(fp,1,'double');

% actual frame data
im = fread(fp,width*height*header.bytes_per_pixel,['*',matlabclass]);
% TODO: handle colorspaces other than RGB8 and MONO8
if ~ismember(lower(header.coding),{'mono8','rgb8'}),
  error('Colorspace %s not yet supported. Only MONO8 and RGB8 allowed.',header.coding);
end
im = reshape(im,[header.ncolors,height,width]);

% store in cache
if isfield(header,'cachedmeans'),
  [~,idxreplace] = min(header.cachedmeans_accesstime);
  header.cachedmeans(:,:,:,idxreplace) = im;
  header.cachedmeans_idx(idxreplace) = meani;
  header.cachedmeans_accesstime(idxreplace) = now;
end
