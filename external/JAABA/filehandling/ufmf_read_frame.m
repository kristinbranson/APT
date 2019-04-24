function [im,header,timestamp,bb,mu] = ufmf_read_frame(header,framei,interruptible)

if nargin < 3,
  interruptible = false;
end

FRAME_CHUNK = 1;

fp = header.fid;

% read from the current location if no frame input
if nargin < 2,
  loc = ftell(fp);
  framei = find(header.frame2file==loc,1);
  if isempty(framei),
    error('Could not find file location %d in frame index',loc);
  end
else
  fseek(fp,header.frame2file(framei),'bof');
end

if interruptible,
  drawnow;
end

% read in the chunk type: 1
chunktype = fread(fp,1,'uchar');
if chunktype ~= FRAME_CHUNK,
  error('Expected chunktype = %d at start of frame, got %d',FRAME_CHUNK,chunktype);
end
% read in timestamp: 8
timestamp = fread(fp,1,'double');
if header.version == 4,
  % number of points: 2
  npts = fread(fp,1,'uint32');
else
  % number of points: 2
  npts = fread(fp,1,'ushort');
end
%fprintf('nforeground boxes = %d\n',npts);

if interruptible,
  drawnow;
end

% sparse-matrix
if header.is_fixed_size,
  bb = fread(fp,npts*2,'uint16');
  bb = reshape(bb,[npts,2]);
  % read sideways
  bb = bb(:,[2,1]);
  data = fread(fp,npts*header.max_width*header.max_height*header.bytes_per_pixel,['*',header.dataclass]);
  % TODO: handle colorspaces other than MONO8 and RGB8
  data = reshape(data,[header.ncolors,npts,header.max_height,header.max_width]);
else
  bb = zeros(npts,4);
  data = cell(1,npts);
  if framei == header.nframes,
    for i = 1:npts,
      bb(i,:) = fread(fp,4,'ushort');
      width = bb(i,4); height = bb(i,3);
      data{i} = fread(fp,width*height*header.bytes_per_pixel,['*',header.dataclass]);
      % TODO: handle colorspaces other than MONO8 and RGB8
      data{i} = reshape(data{i},[header.ncolors,height,width]);
    end
  else
    cache = fread(fp,(header.frame2file(framei+1)-header.frame2file(framei)+1)*header.bytes_per_pixel,...
      ['*',header.dataclass]);
    cacheidx=1;
    for i = 1:npts,
      %bb(i,:) = fread(fp,4,'ushort');
      tmp=double(cache(cacheidx:(cacheidx+7)));
      bb(i,:)=tmp(1:2:7)+256*tmp(2:2:8);
      width = bb(i,4); height = bb(i,3);
      data{i} = cache((cacheidx+8):(cacheidx+7+width*height*header.bytes_per_pixel));
      cacheidx=cacheidx+8+width*height*header.bytes_per_pixel;
      %width = bb(i,4); height = bb(i,3);
      %data{i} = fread(fp,width*height*header.bytes_per_pixel,['*',header.dataclass]);
      % TODO: handle colorspaces other than MONO8 and RGB8
      data{i} = reshape(data{i},[header.ncolors,height,width]);
    end
  end
  % images are read sideways
  bb = bb(:,[2,1,4,3]);
end
% matlab indexing
bb(:,1:2) = bb(:,1:2)+1;

if interruptible,
  drawnow;
end

% read in the mean image
[mu,header] = ufmf_read_mean(header,'framei',framei,'dopermute',false);
if ~strcmp(header.dataclass,header.meandataclass),
  mu = cast(mu,header.dataclass);
end
im = mu;

if interruptible,
  drawnow;
end

if header.is_fixed_size,
  % sparse image
  if header.max_height == 1 && header.max_width == 1,
    tmp = false(header.nr,header.nc);
    tmp(sub2ind(size(tmp),bb(:,2),bb(:,1))) = true;
    im(:,tmp) = data;
  else
    for i = 1:npts,
      im(:,bb(i,2):bb(i,2)+max_height-1,bb(i,1):bb(i,1)+max_width-1) = data(:,i,:,:);
      if interruptible && mod(i,50) == 0,
        drawnow;
      end
    end
  end
else
  % boxes
  for i = 1:npts,
    im(:,bb(i,2):bb(i,2)+bb(i,4)-1,bb(i,1):bb(i,1)+bb(i,3)-1) = data{i};
    if interruptible && mod(i,50) == 0,
      drawnow;
    end
  end
end

if interruptible,
  drawnow;
end

im = permute(im,[3,2,1]);
if nargout >= 5,
  mu = permute(mu,[3,2,1]);
end
%im = permute(im,[2,3,1]);
%mu = permute(mu,[2,3,1]);
