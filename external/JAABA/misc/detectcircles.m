% [r,x,y,params] = detectcircles(im,...)
% inputs:
% im: image to detect circles in
% parameters:
% cannythresh: thresholds for canny edge detection
% cannysigma: standard deviation for canny edge detection
% binedgesa: bin edges for x-coordinate of center
% bincentersb: bin centers for y-coordinate of center
% bincentersr: bin centers for radius
% peaksnhoodsize: size of the suppression neighborhood when searching for
% peaks in the circle-ness. this is the neighborhood around each peak
% that is set to zero after the peak is identified. 
% peaksthreshold: minimum circleness value. 
% maxncircles: maximum number of circles to detect

function [r,x,y,score,params] = detectcircles(im,varargin)

[cannythresh,cannysigma,binedgesa,bincentersb,bincentersr,...
 peaksnhoodsize,peaksthreshold,maxncircles,doedgedetect] = ...
    myparse(varargin,'cannythresh',[],'cannysigma',1,...
            'binedgesa',[],'bincentersb',[],'bincentersr',[],...
            'peaksnhoodsize',[],'peaksthreshold',[],...
            'maxncircles',1,'doedgedetect',true);

[nr,nc] = size(im);

% choose bins for estimating circle size
nbinsdefault = 10;
mina = 1;
maxa = nc;
minb = 1;
maxb = nr;
minr = min(nr,nc)/4;
maxr = min(nr,nc)/2;
peakratio = 50;
if isempty(binedgesa),
  binedgesa = linspace(mina,maxa,nbinsdefault+1);
end;
bincentersa = (binedgesa(1:end-1)+binedgesa(2:end))/2;
if isempty(bincentersb),
  binedgesb = linspace(minb,maxb,nbinsdefault+1);
  bincentersb = (binedgesb(1:end-1)+binedgesb(2:end))/2;
end;
if isempty(bincentersr),
  binedgesr = linspace(minr,maxr/2,nbinsdefault+1);
  bincentersr = (binedgesr(1:end-1)+binedgesr(2:end))/2;
end;
% set peak neighborhood size
if isempty(peaksnhoodsize),
  peaksnhoodsize = [(length(binedgesa)-1)/peakratio,...
                    length(bincentersb)/peakratio,...
                    length(bincentersr)/peakratio];
  peaksnhoodsize = max(2*ceil(peaksnhoodsize/2) + 1, 1); 
end;

% detect edges in image
if doedgedetect,
  [bw,cannythresh] = edge(im,'canny',cannythresh,cannysigma);
  [r,c] = find(bw);
else
  bw = im;
  [r,c] = find(bw);
end

% find circles using the hough transform
acc = houghcircles(c,r,binedgesa,bincentersb,bincentersr);

% set threshold
if isempty(peaksthreshold),
  peaksthreshold = max(acc(:))/2;
end;

peaks = houghcirclepeaks(acc,maxncircles,'threshold',peaksthreshold,...
  'nhoodsize',peaksnhoodsize);

%bigcircle = argmax(peaks(:,3));

x = bincentersa(peaks(:,1));
y = bincentersb(peaks(:,2));
r = bincentersr(peaks(:,3));
score = acc(sub2ind(size(acc),peaks(:,1),peaks(:,2),peaks(:,3)));

params = {'cannythresh',cannythresh,'cannysigma',cannysigma,...
          'binedgesa',binedgesa,'bincentersb',bincentersb,...
          'bincentersr',bincentersr,'peaksnhoodsize',peaksnhoodsize,...
          'peaksthreshold',peaksthreshold};
