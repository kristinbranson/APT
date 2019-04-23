function [im,header,timestamp,bb,mu] = mmf_read_frame(header,framei)

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

[mu,header] = mmf_read_mean(header,'framei',framei);

im = mu;
im_idcode = fread(fp,1,'ulong'); %#ok<NASGU>
im_headersize = fread(fp,1,'int');
%im_depth = fread(fp,1,'int');
%im_nchannels = fread(fp,1,'int');
fseek(fp,8,'cof');
im_numblocks = fread(fp,1,'int');
    
% seek to the end of the header
fseek(fp,header.frame2file(framei)+im_headersize,'bof');

bb = nan(im_numblocks,4);
%for j = 1:im_numblocks,
  block_rect = fread(fp,4,'int');
  j=1;
while numel(block_rect) == 4 && block_rect(1)>=0
  %if block_rect(1)<0
      %disp(['skipping blocks ',num2str(framei),' ',mat2str(block_rect),sprintf(' block %d / %d',j,im_numblocks)])
      %break;
  %end
  bb(j,[2,1,4,3]) = block_rect;
  if block_rect(3)>header.height || block_rect(4) > header.width || any(block_rect<=0),
    break;
  end

  % then interlaced row ordered image data
  block_data = fread(fp,block_rect(3)*block_rect(4),'*uint8');
  block_data = reshape(block_data,[block_rect(3),block_rect(4)]);
  im(block_rect(1)+1:block_rect(1)+block_rect(3),block_rect(2)+1:block_rect(2)+block_rect(4)) = block_data;
  [block_rect,count] = fread(fp,4,'int');
  if count < 4,
    break;
  end
  j=j+1;
end
%disp([num2str(framei),' ',mat2str(block_rect),sprintf(' block %d / %d',j-1,im_numblocks)])
timestamp = nan;