function [H0,tfuse] = typicalImHist(I,varargin)
% Compute typical imhist of set of images
%
% I: [N] cell array of identically-sized images
%
% H0: [nbin] 'typical' imhist for I.
% tfuse: [N] logical indicating whether each image in I was used in 
%   computing H

[nbin,wbObj,incPrctile] = myparse(varargin,...
  'nbin',256,...
  'wbObj',[],... % WaitBarWithCancel. If canceled, H0 and tfuse are indeterminate
  'incPrctile',75 ... % include images in I with average intensity at this 
                  ... % prctile or above (prefer brighter images)
  );

tfWB = ~isempty(wbObj);

if ~iscell(I) 
  error('typicalImHist:I','Input I must be a cell array of images.');
end
N = numel(I);
if N==0
  warning('typicalImHist:I','Empty input image array.');
  H0 = 1/nbin*ones(nbin,1);
  tfuse = false(N,1);
  return;
end

imNpx = cellfun(@numel,I);
if ~all(imNpx==imNpx(1))
  error('typicalImHist:I',...
    'Input images must all have the same number of elements.');
end
imNpx = imNpx(1);
H = nan(nbin,N);
mu = nan(1,N);
loc = [];
if tfWB
  wbObj.startPeriod('Performing histogram equalization');
  oc = onCleanup(@()wbObj.endPeriod);
end
for i = 1:N
  if tfWB
    tfCancel = wbObj.updateFrac(i/N);
    if tfCancel
      H0 = nan(nbin,1);
      tfuse = false(N,1);      
      return
    end
  end
  
  im = I{i};
  [H(:,i),loctmp] = imhist(im,nbin);
  if i==1
    loc = loctmp;
  else
    assert(isequal(loctmp,loc));
  end
  mu(i) = mean(im(:));
end
% normalize to brighter rather than dimmer movies
tfuse = mu >= prctile(mu,incPrctile);
fprintf('using %d frames to form H0:\n',nnz(tfuse));
H0 = median(H(:,tfuse),2);
H0 = H0/sum(H0)*imNpx;