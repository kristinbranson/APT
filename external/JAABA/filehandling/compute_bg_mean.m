function [mu,sig] = compute_bg_mean(readframe,nframes,varargin)
  
[bgstartframe,bgendframe,bgnframes,roi,verbose] = ...
    myparse(varargin,'bgstartframe',1,'bgendframe',nframes,...
            'bgnframes',200,'roi',[],'verbose',1);
          
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

mu = zeros(nr1,nc1);
sig = zeros(nr1,nc1);

for ii = 1:bgnframes,
  tt = framessample(ii);
  im1 = double(rgb2gray(readframe(tt)));
  mu = mu + im1(roi(3):roi(4),roi(1):roi(2));
  sig = sig + mu.^2;
end

mu = mu / bgnframes;
sig = sqrt(sig / bgnframes - mu.^2);

