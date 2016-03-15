function [tdR,tdL,tdTop] = splitDataRomain(td)

XBOUNDARY = 600;
YBOUNDARY = 275;
LBLS = {[1:18 55] [19:36 56] [37:54 57]}; % top, left, right views

n = td.N;
npts = 19; % pts per view
D = npts*2;
assert(isequal(sort([LBLS{1} LBLS{2} LBLS{3}]),1:3*npts));

I = cell(n,3); % IsTr{iF,i} is training image for frame frms(iF), index i (i in 1..3)
pGT = nan(n,D,3); % pGT(:,:,1)
for iTrl = 1:n
  im = td.I{iTrl};  
  im1 = im(:,1:XBOUNDARY);
  im2 = im(1:YBOUNDARY,XBOUNDARY+1:end);
  im3 = im(YBOUNDARY+1:end,XBOUNDARY+1:end);
  I{iTrl,1} = im1;
  I{iTrl,2} = im2;
  I{iTrl,3} = im3;
  
  p = td.pGT(iTrl,:);
  xy = Shape.vec2xy(p);
  assert(isequal(size(xy),[npts*3 2]));

  xy1 = xy(LBLS{1},:);
  xy2 = xy(LBLS{2},:);
  xy3 = xy(LBLS{3},:);  
  pGT(iTrl,:,1) = Shape.xy2vec(xy1);
  xy2(:,1) = xy2(:,1)-XBOUNDARY;
  pGT(iTrl,:,2) = Shape.xy2vec(xy2);
  xy3(:,1) = xy3(:,1)-XBOUNDARY;
  xy3(:,2) = xy3(:,2)-YBOUNDARY;
  pGT(iTrl,:,3) = Shape.xy2vec(xy3);
end

bb = cell(1,3);
for iView = 1:3
  sz = cellfun(@(x)size(x'),I(:,iView),'uni',0);
  bb{iView} = cellfun(@(x)[[1 1] x],sz,'uni',0);
end

% md
MD = cell(1,3);
for iView = 1:3
  tv = cell(n,1);
  nLblInf = nan(n,1);
  for iTrl = 1:n
    [tf,loc] = ismember(td.MD.tagvec{iTrl},LBLS{iView});
    tv{iTrl} = loc(tf);
    
    xy = Shape.vec2xy(pGT(iTrl,:,iView));
    nLblInf(iTrl) = sum(any(isinf(xy),2));
  end
  
  MDnew = td.MD;
  MDnew.tagvec = tv;
  MDnew.nTag = cellfun(@numel,MDnew.tagvec);
  MDnew.nLblInf = nLblInf;
  assert(all(MDnew.nLblNaN==0));  
  
  MD{iView} = MDnew;
end
  
tdTop = CPRData(I(:,1),pGT(:,:,1),bb{1},MD{1});
tdL = CPRData(I(:,2),pGT(:,:,2),bb{2},MD{2});
tdR = CPRData(I(:,3),pGT(:,:,3),bb{3},MD{3});
