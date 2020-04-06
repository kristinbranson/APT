function [im,header] = mmf_read_mean(header,varargin)

[meani,framei] = myparse(varargin,'meani',[],'framei',[]);
fp = header.fid;

meani = [meani,header.frame2mean(framei)];

if isempty(meani),
  [im,header] = mmf_read_mean_helper(fp,header);
else
  if length(meani) > 1,
    im = zeros([header.nr,header.nc,length(meani)],'uint8');
  end
  if isfield(header,'cachedmeans_idx'),
    [idx,cachei] = ismember(meani,header.cachedmeans_idx);
  else
    idx = false(size(meani));
  end
  for i = find(idx),
    im(:,:,i) = header.cachedmeans(:,:,cachei(i));
    header.cachedmeans_accesstime(i) = now;
  end
  for i = find(~idx),
    fseek(fp,header.mean2file(meani(i)),'bof');
    [im(:,:,i),header] = mmf_read_mean_helper(fp,header);
  end
end

function [im,header,timestamp] = mmf_read_mean_helper(fp,header)

loc = ftell(fp);
meani = find(header.mean2file == loc,1);
if isempty(meani),
  error('Could not find current file location in mean2file index');
end
  
im = fread(fp,header.bkgdim_imagesize,'uint8=>uint8');
im = reshape(im,header.bkgdim_widthstep,header.width);
im = im(1:header.height,:);

% store in cache
if isfield(header,'cachedmeans'),
  [~,idxreplace] = min(header.cachedmeans_accesstime);
  header.cachedmeans(:,:,idxreplace) = im;
  header.cachedmeans_idx(idxreplace) = meani;
  header.cachedmeans_accesstime(idxreplace) = now;
end

timestamp = nan;
