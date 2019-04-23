function [med,mad] = compute_bg_med(readframe,nframes,varargin)

[bgstartframe,bgendframe,bgnframes,roi,maxmem,verbose] = ...
    myparse(varargin,'bgstartframe',1,'bgendframe',nframes,...
            'bgnframes',200,'roi',[],...
            'maxmem',2^20*50,'verbose',1);

if isempty(roi),
  im = readframe(bgstartframe);
  nr = size(im,1);
  nc = size(im,2);
  roi = [1,nc,1,nr];
end

% frames to read in
framessample = round(linspace(bgstartframe,bgendframe,bgnframes));

% size of ROI
nr1 = roi(4)-roi(3)+1;
nc1 = roi(2)-roi(1)+1;
  
med = zeros(nr1,nc1);
mad = zeros(nr1,nc1);
  
% we don't want to store more than maxmem bytes in memory at any one time
ncperiter = min(floor(maxmem / (bgnframes * nr1 * 8) ),nc1);
% starts of new blocks to read
coff = roi(1)-1;
cstart = 1:ncperiter:nc1+1;
if cstart(end) < nc1+1,
  cstart(end+1) = nc1+1;
end
niters = length(cstart)-1;

imbuf = zeros(nr1,ncperiter);

for iter = 1:niters,
  
  c0 = cstart(iter);
  c1 = cstart(iter+1)-1;
  nccurr = c1 - c0 + 1;
  
  if verbose >= 2,
    fprintf('Background modeling iteration %d/%d: processing columns %d:%d\n',iter,niters,c0,c1);
  end
  
  for ii = 1:bgnframes,
    tt = framessample(ii);
    im1 = double(rgb2gray(readframe(tt)));
    imbuf(:,1:nccurr,ii) = im1(roi(3):roi(4),coff+c0:coff+c1);
  end
  
  med(:,c0:c1) = median(imbuf(:,1:nccurr,:),3);
  for ii = 1:bgnframes,
    imbuf(:,1:nccurr,ii) = imabsdiff(imbuf(:,1:nccurr,ii),med(:,c0:c1));
  end
  mad(:,c0:c1) = median(imbuf(:,1:nccurr,:),3);    
end

mad = double(mad)*1.482602;

