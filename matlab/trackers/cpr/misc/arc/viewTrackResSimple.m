function viewTrackResSimple(movname,resname)

fprintf('Loading results...\n');
res = load(resname);

lObj = Labeler;
lObj.projLoadGUI(lblname);

%% new moviereader, labelCore
fprintf('Setting new MovieReader, LabelCore...\n');

nf = lObj.nframes;
nlf = td.NFullyLabeled;

assert(numel(td.I)==nf);
mr = MovieReaderImStack;
mr.open(td.I);
lObj.movieReader = mr;

[~,D] = size(td.pGT);
xyGT = reshape(td.pGT,[nf D/2 2]);

pTstT = res.pTstT(:,:,:,end);
pTstT = permute(pTstT,[1 3 2]);
assert(size(pTstT,2)==D);
[nTrlPTstT,~,nrep] = size(pTstT);
tfOnlyLabeledTracked = nTrlPTstT==nlf;
if tfOnlyLabeledTracked
  fprintf(1,'pTstT only includes labeled frames.\n');
  xyTstT = reshape(pTstT,[nlf D/2 2 nrep]);
  xyTstTPadded = nan(nf,D/2,2,nrep);
  xyTstTPadded(td.isFullyLabeled,:,:,:) = xyTstT;
else
  fprintf(1,'pTstT contains all frames.\n');
  clear xyTstT
  xyTstTPadded = reshape(pTstT,[nf D/2 2 nrep]);
end

pTstTRed = res.pTstTRed(:,:,end);
if tfOnlyLabeledTracked
  xyTstTRed = reshape(pTstTRed,[nlf D/2 2]);
  xyTstTRedPadded = nan(nf,D/2,2);
  xyTstTRedPadded(td.isFullyLabeled,:,:) = xyTstTRed;
else
  clear xyTstTRed
  xyTstTRedPadded = reshape(pTstTRed,[nf D/2 2]);
end

lc = LabelCoreCPRView(lObj);
lc.setPs(xyGT,xyTstTPadded,xyTstTRedPadded);
delete(lObj.lblCore);
lObj.lblCore = lc;
lc.init(lObj.nLabelPoints,lObj.labelPointsPlotInfo);
lObj.setFrameGUI(1);


%% plot error by landmark over frames
assert(isequal(size(xyGT),size(xyTstTRedPadded)));
delta = sqrt(sum((xyGT-xyTstTRedPadded).^2,3)); % [nf x npt]
figure;
tfLbled = td.isFullyLabeled;
plot(find(tfLbled),delta(tfLbled,4:7),'-.');
legend({'4' '5' '6' '7'});
grid on;

%% Susp
tfLbled = td.isFullyLabeled;
if tfOnlyLabeledTracked
  [d,dav] = Shape.distP(td.pGT(tfLbled,:),reshape(pTstT,[nlf D nrep]));
else
  [d,dav] = Shape.distP(td.pGT(tfLbled,:),reshape(pTstT(tfLbled,:,:),[nlf D nrep]));  
end
d = d(:,4:7,:);
d = squeeze(mean(d,2)); % npt x nrep
d = mean(d,2);
dPad = zeros(nf,1);
dPad(tfLbled) = d;
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
    hPts(i,j).XData = xyGT(f,j);
    hPts(i,j).YData = xyGT(f,j+7);
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

%% View Ftrs
iTrl = 258; % = frame1286, worst susp
muFtrDist = Shape.vizRepsOverTime(td.I(td.isFullyLabeled),res.pTstT,...
  iTrl,tr.regModel.model,'nr',2,'nc',2,'regs',tr.regModel.regs);

%% View Fern stuff
regs = tr.regModel.regs;
thrs1 = nan(0,1);
fernD = nan(0,7);
for iReg = 1:numel(regs)
  ri = regs(iReg).regInfo;
  for iRI = 1:numel(ri)
    thrs1(end+1,1) = ri{iRI}.thrs(1);
    ys = reshape(ri{iRI}.ysFern,[32 7 2]);
    tmp = sqrt(sum(ys.^2,3)); % [32x7], dist for each pt
    fernD(end+1,:) = mean(tmp,1);
  end
end
    
figure;
x = 1:5000;
plot(x,thrs1);
figure;
plot(x,fernD);