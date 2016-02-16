LBLNAME = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\allen cleaned lbls\150730_2_006_2_C001H001S0001.lbl';
TDFILE = 'f:\cpr\data\jan\td@150730_2_006_2_all@pp_lbled@0129';
RESFILE = 'f:\DropBoxNEW\Dropbox\Tracking_KAJ\track.results\11exps@@for_150730_02_006_02@iTrn@lotsa1__11exps@@for_150730_02_006_02@iTst__0205T1608\res.mat';

%% load results
fprintf('Loading results...\n');
res = load(RESFILE);

%% 
fprintf('Loading TD...\n');
td = load(TDFILE);
td = td.td;

%% open Labeler
lObj = Labeler;
lObj.projLoad(LBLNAME);

%% new moviereader, labelCore
fprintf('Setting new MovieReader, LabelCore...\n');

nf = lObj.nframes;
nlf = td.NFullyLabeled;

assert(numel(td.I)==nf);
mr = MovieReaderImStack;
mr.open(td.I);
lObj.movieReader = mr;

[~,D] = size(td.pGT);
pGT = reshape(td.pGT,[nf D/2 2]);

pTstT = res.pTstT(:,:,:,end);
pTstT = permute(pTstT,[1 3 2]);
assert(size(pTstT,1)==nlf);
assert(size(pTstT,2)==D);
[~,~,nrep] = size(pTstT);
xyTstT = reshape(pTstT,[nlf D/2 2 nrep]);
xyTstTPadded = nan(nf,D/2,2,nrep);
xyTstTPadded(td.isFullyLabeled,:,:,:) = xyTstT;

pTstTRed = res.pTstTRed(:,:,end);
xyTstTRed = reshape(pTstTRed,[nlf D/2 2]);
xyTstTRedPadded = nan(nf,D/2,2);
xyTstTRedPadded(td.isFullyLabeled,:,:) = xyTstTRed;

lc = LabelCoreCPRView(lObj);
lc.setPs(pGT,xyTstTPadded,xyTstTRedPadded);
delete(lObj.lblCore);
lObj.lblCore = lc;
lc.init(lObj.nLabelPoints,lObj.labelPointsPlotInfo);
lObj.setFrame(1);

%% Susp
[d,dav] = Shape.distP(td.pGT(td.isFullyLabeled,:),...
  reshape(pTstT,[nlf D nrep]));
d = d(:,4:7,:);
d = squeeze(mean(d,2)); % npt x nrep
d = mean(d,2);
dPad = zeros(nf,1);
dPad(td.isFullyLabeled) = d;
lObj.setSuspScore({dPad});


%% View: time-lapse of GT
figure;
ax = axes;
imshow(td.I{1});
hold(ax,'on');
colors = jet(7);
NSHOW = 20;
PAUSEINT = .1;
hPts = gobjects(NSHOW,4);
for i=1:NSHOW
  for j=4:7
    hPts(i,j) = plot(nan,nan,'o','MarkerFaceColor',colors(j,:));
  end
end
for f=1:5:3661
  i=mod(f,20)+1; 
  for j=4:7
    hPts(i,j).XData = pGT(f,j);
    hPts(i,j).YData = pGT(f,j+7);
  end
  title(num2str(f));
  pause(PAUSEINT);
end

%% d (pGT to pred), pred-spread, pred-jump
d = Shape.distP(td.pGT(td.isFullyLabeled,:),pTstTRed);
d = d(:,4:7); % nlfx4

predSD = nan(nlf,4); 
for iTrl = 1:nlf
for iPt = 1:4
  xy = squeeze(xyTstT(iTrl,iPt+3,:,:)); % [2xnrep]
  xymu = mean(xy,2);
  d2 = sum(bsxfun(@minus,xy,xymu).^2,1); % [1xnrep] squared dists from centroid
  predSD(iTrl,iPt) = mean(d2);
end
end

%fprintf(2,'Assuming labeled frames are first n GT frames.\n');
predJMP = nan(nlf,4);
for iTrl = 2:nlf
for iPt = 1:4
  xy1 = squeeze(xyTstTRed(iTrl,iPt+3,:)); % [2x1]
  xy0 = squeeze(xyTstTRed(iTrl-1,iPt+3,:)); % [2x1]
  predJMP(iTrl,iPt) = sum((xy1-xy0).^2);
end
end

figure;
ax = createsubplots(2,4);
ax = reshape(ax,[2 4]);
x = 1+(0:nlf-1)*5;
for iPt = 1:4
  plot(ax(1,iPt),x,5*d(:,iPt),x,predSD(:,iPt),x,predJMP(:,iPt));
  if iPt==1
    legend(ax(1,iPt),'dmu','predSD','predJMP');
    title(ax(1,iPt),num2str(iPt+3));
  end
  axes(ax(2,iPt));
  scatter(5*d(:,iPt),predSD(:,iPt));
end


