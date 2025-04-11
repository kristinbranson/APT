%%
ROOTDIR = 'f:\Dropbox\MultiViewFlyLegTracking\multiview labeling';
%assert(strcmp(pwd,ROOTDIR));
%LBLF2 = 'f:\Dropbox\MultiViewFlyLegTracking\trackingApril28-14-53\20160428T145316_allen.lbl';

LBL = 'romainJun22NewLabels.lbl';
LBL = fullfile(ROOTDIR,LBL);

CRIG = 'crig2Optimized_calibjun2916_roiTrackingJun22_20160810_AllExtAllInt.mat';
CRIG = fullfile(ROOTDIR,CRIG);

NVIEW = 3;
NREALPT = 57/3;

%%
lbl = load(LBL,'-mat');
crig2 = load(CRIG,'-mat');
%crig2 = crig2.crig2Mod;
crig2 = crig2.crig2AllExtAllInt;

%%
lpos = lbl.labeledpos{1};
lpostag = lbl.labeledpostag{1};
nfrm = size(lpos,3);
lpos = reshape(lpos,[NREALPT NVIEW 2 nfrm]);
lpostag = reshape(lpostag,[NREALPT NVIEW nfrm]);

%% Generate MD table
% 
% Fields
% frm
% npts2VwLblNO. scalar, number of pts that have >= 2 views labeled (nonOcc)
% ipts2VwLblNO. vector with npts2VwLblNO els. pt indices.
% iVwsLblNO. cell vector with npts2VwLblNO els. view indices for each pt in
%   ipts2VwLblNO.
frm = nan(0,1);
npts2VwLblNO = nan(0,1);
ipts2VwLblNO = cell(0,1);
iVwsLblNO = cell(0,1);
iVwsLblNOCode = cell(0,1);
% yL = nan(0,2); % GT pts: (row,col) cropped coords in L view
% yR = nan(0,2);
% yB = nan(0,2);
VIEWCODES = 'lrb';

frmAny = nan(0,1);
npts2VwLblAny = nan(0,1);
ipts2VwLblAny = cell(0,1);
iVwsLblAny = cell(0,1);
iVwsLblAnyCode = cell(0,1);
for f=1:nfrm
  
  iVwLbledNonOccPt = cell(NREALPT,1);
  tf2VwLbledNonOcc = false(NREALPT,1);
  iVwLbledAnyPt = cell(NREALPT,1);
  tf2VwLbledAny = false(NREALPT,1);
  for ipt = 1:NREALPT
    lposptfrm = squeeze(lpos(ipt,:,:,f));
    ltagptfrm = squeeze(lpostag(ipt,:,f));
    ltagptfrm = ltagptfrm(:);
    assert(isequal(size(lposptfrm),[NVIEW 2]));
    assert(isequal(size(ltagptfrm),[NVIEW 1]));
    tfVwLbled = ~any(isnan(lposptfrm),2);
    tfVwNotOcc = cellfun(@isempty,ltagptfrm);    
    iVwLbledNonOccPt{ipt} = find(tfVwLbled & tfVwNotOcc);      
    tf2VwLbledNonOcc(ipt) = numel(iVwLbledNonOccPt{ipt})>=2;
    
    iVwLbledAnyPt{ipt} = find(tfVwLbled);
    tf2VwLbledAny(ipt) = nnz(tfVwLbled)>=2;
  end

  if any(tf2VwLbledNonOcc)
    frm(end+1,1) = f;
    npts2VwLblNO(end+1,1) = nnz(tf2VwLbledNonOcc);
    ipts2VwLblNO{end+1,1} = find(tf2VwLbledNonOcc);
    iVwsLblNO{end+1,1} = iVwLbledNonOccPt(tf2VwLbledNonOcc);
    iVwsLblNOCode{end+1,1} = cellfun(@(x)VIEWCODES(x),...
      iVwLbledNonOccPt(tf2VwLbledNonOcc),'uni',0);
  end
  
  if any(tf2VwLbledAny)
    frmAny(end+1,1) = f;
    npts2VwLblAny(end+1,1) = nnz(tf2VwLbledAny);
    ipts2VwLblAny{end+1,1} = find(tf2VwLbledAny);
    iVwsLblAny{end+1,1} = iVwLbledAnyPt(tf2VwLbledAny);
    iVwsLblAnyCode{end+1,1} = cellfun(@(x)VIEWCODES(x),...
      iVwLbledAnyPt(tf2VwLbledAny),'uni',0);    
  end
  
%   tf2VwLbledNonOcc(end+1,1) = nnz(tfVwLbled & tfVwNotOcc)>=2;
%   yL(end+1,:) = lpos(ipt,1,[2 1],f);
%   yR(end+1,:) = lpos(ipt,2,[2 1],f);
%   yB(end+1,:) = lpos(ipt,3,[2 1],f);  
end

nPtsLRCode = cellfun(@(x)nnz(strcmp('lr',x)),iVwsLblNOCode);
tFrmPts = table(frm,npts2VwLblNO,ipts2VwLblNO,iVwsLblNO,iVwsLblNOCode,nPtsLRCode);
% nGood = numel(iPtGood);
% fprintf('Found %d labeled pts.\n',nGood);
tFrmPtsAny = table(frmAny,npts2VwLblAny,ipts2VwLblAny,iVwsLblAny,iVwsLblAnyCode);
tFrmPtsAny.Properties.VariableNames{'frmAny'} = 'frm';
%% KB: SKIP AHEAD TO "BEGIN 4-CORNER TRACKING"

%% Look at LR codes
% for i=1:size(tFrmPts,1)
%   codes = tFrmPts.iVwsLblNOCode{i};
%   for iV = 1:numel(codes)
%     if strcmp(codes{iV},'lr')
%       fprintf(1,'%d: %d\n',tFrmPts.frm(i),tFrmPts.ipts2VwLblNO{i}(iV));
%     end
%   end
% end


%%
%%%%%%%%%%%%%%%%%%
%% Reconstruct/err stats for rows with 19 2-view-NO labeled pts
%%%%%%%%%%%%%%%%%%
%
% For points labeled in all three views ('lrb'):
%  * Use each viewpair to recon/project in 3rd view and compute error.
%  * Recon 3D pt.
% For all points labeled in only two views:
%  * Recon 3D pt.
%
% New fields:
%  * X [3x19]. 3D recon pt in certain frame, say 'l'.
%  * errReconL. [npts2VwlblNOx1]. For points with 'lrb'. L2 err in L view (recon vs gt)
%  * errReconR. etc
%  * errReconB.

t19 = tFrmPts(tFrmPts.npts2VwLblNO==19,:);
nRows = size(t19,1);
XL = cell(nRows,1);
XLlr = cell(nRows,1);
errReconL = nan(nRows,19);
errReconR = nan(nRows,19);
errReconB = nan(nRows,19);
errReconL_lr = nan(nRows,19);
errReconR_lr = nan(nRows,19);
errReconB_lr = nan(nRows,19);
for iRow = 1:nRows
  frm = t19.frm(iRow);
  XLrow = nan(3,19);
  XLlrrow = nan(3,19);
  for iPt = 1:19
    code = t19.iVwsLblNOCode{iRow}{iPt};
    
    lposPt = squeeze(lpos(iPt,:,:,frm));
    assert(isequal(size(lposPt),[3 2]));
    yL = lposPt(1,[2 1]);
    yR = lposPt(2,[2 1]);
    yB = lposPt(3,[2 1]);
    switch code
      case 'lr'
        %assert(all(isnan(yB(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateCropped(yL,yR,'L','R');        
      case 'lb'
        %assert(all(isnan(yR(:))));
        XLrow(:,iPt) = crig2.stereoTriangulateLB(yL,yB);        
      case 'rb'
        %assert(all(isnan(yL(:))));
        XBbr = crig2.stereoTriangulateBR(yB,yR);
        XLrow(:,iPt) = crig2.camxform(XBbr,'bl');
      case 'lrb'
        [~,~,~,...
          errReconL(iRow,iPt),errReconR(iRow,iPt),errReconB(iRow,iPt),...
          ~,~,~,...
          XLrow(:,iPt),~,~,...
          errReconL_lr(iRow,iPt),errReconR_lr(iRow,iPt),errReconB_lr(iRow,iPt),...
          XLlrrow(:,iPt)] = ...
            crig2.calibRoundTripFull(yL,yR,yB);
    end
  end
  XL{iRow} = XLrow;
  XLlr{iRow} = XLlrrow;
end

tAug = table(XL,errReconL,errReconR,errReconB,...
  errReconL_lr,errReconR_lr,errReconB_lr,XLlr);
t19aug = [t19 tAug];

%% Make an expanded frm/pt err browsing table
s19expanded = struct(...
  'frm',cell(0,1),...
  'ipt',[],...
  'code',[],...
  'XL',[],...
  'errReconL',[],...
  'errReconR',[],...
  'errReconB',[],...
  'errReconL_lr',[],...
  'errReconR_lr',[],...
  'errReconB_lr',[],...
  'XLlr',[]);
%%
nRows = size(t19aug,1);
ERRFLDS = {'errReconL' 'errReconR' 'errReconB' 'errReconL_lr' 'errReconR_lr' 'errReconB_lr'};
for iRow=1:nRows
  assert(t19aug.npts2VwLblNO(iRow)==19);
  assert(isequal(t19aug.ipts2VwLblNO{iRow},(1:19)'));
  frm = t19aug.frm(iRow);
  for iPt = 1:19
    s19expanded(end+1,1).frm = frm;
    s19expanded(end,1).ipt = iPt;
    s19expanded(end,1).code = t19aug.iVwsLblNOCode{iRow}{iPt};
    s19expanded(end,1).XL = t19aug.XL{iRow}(:,iPt)';
    s19expanded(end,1).XLlr = t19aug.XLlr{iRow}(:,iPt)';
    for f=ERRFLDS,f=f{1}; %#ok<FXSET>
      s19expanded(end,1).(f) = t19aug.(f)(iRow,iPt);
    end
  end
end
t19expanded = struct2table(s19expanded);
t19expandedLRB = t19expanded(strcmp(t19expanded.code,'lrb'),:);
%% 
hFig = figure('windowstyle','docked');
for f = ERRFLDS,f=f{1};
  clf(hFig);
  z = t19expandedLRB.(f);
  hist(z,50);
  iptsbig = unique(t19expandedLRB.ipt(z>20));
  fprintf('\n%s:\n',f);
  disp(iptsbig);
  input('hk');
end

%%
t19expandedLRB = t19expandedLRB(t19expandedLRB.ipt~=19,:);

%%%%%%%%%%%%%%%%%%
%% END Reconstruct/err stats for rows with 19 2-view-NO labeled pts
%%%%%%%%%%%%%%%%%%

%% 
%%%%%%%%%%%%%%%%%
%% BEGIN 4-corner tracking 20161102 (now not 4-corner, all legs)
%%%%%%%%%%%%%%%%%

iPtLegs = [...
  1 7 13; % LF
  2 8 14; % LM
  3 9 15; % LH
  4 10 16; % RF
  5 11 17; % RM
  6 12 18]; % RH

iLegsUse = 1:6;
iPtLegsAllUsed = iPtLegs(iLegsUse,:);

%% find frames with 4-corners 2-view-NO labeled
N = size(tFrmPts,1);
%tfLegsLbled = arrayfun(@(x)all(ismember(iPtLegsAllUsed(:),tFrmPts.ipts2VwLblNO{x})),(1:N)');
tfLegsLbledBinc = false(N,1);
for i=1:N  
  [tf,loc] = ismember(iPtLegsAllUsed(:),tFrmPts.ipts2VwLblNO{i});
  if all(tf)
    vwCodes = tFrmPts.iVwsLblNOCode{i};
    vwCodesLegs = vwCodes(loc);
    tfLegsLbledBinc(i) = all(cellfun(@(x)any(x=='b'),vwCodesLegs));
  end
end
tMFP = tFrmPts(tfLegsLbledBinc,:);

% tMFP contains all rows for frames labeled i) in at least 2 views (with 
% non-Occluded labels), where one of the views is the Bottom (eg the Bottom
% must be labeled)

%% find frames with legs 2-view labeled (occlusion OK), bot must have label
N = size(tFrmPtsAny,1);
tfLegsLbledBinc = false(N,1);
for i=1:N
  [tf,loc] = ismember(iPtLegsAllUsed(:),tFrmPtsAny.ipts2VwLblAny{i});
  if all(tf)
    vwCodes = tFrmPtsAny.iVwsLblAnyCode{i};
    vwCodesLegs = vwCodes(loc);
    tfLegsLbledBinc(i) = all(cellfun(@(x)any(x=='b'),vwCodesLegs));
    assert(all(cellfun(@(x)numel(x)>=2,vwCodesLegs)));
  end
end
tMFPAny = tFrmPtsAny(tfLegsLbledBinc,:);

% tMFPAny contains all rows for frames labeled in at least 2 views
% (occluded or not), where the Bottom is labeled.

%%
tMFPTrn = tMFPAny;

mfa = lbl.movieFilesAll;
% KB: modify regexprep for your local filesystem
mfa = regexprep(mfa,'C:\\Users\\nielsone\\Dropbox \(HHMI\)','f:\\Dropbox');
movs = repmat(mfa,size(tMFPTrn,1),1);
movs = struct('movs',{movs});
movs = struct2table(movs);
tMFPTrn = [tMFPTrn movs];

%%
[I,pGT3d,bboxes,pGt3dRCerr,tMFPout] = rfCompileData3D(...
  tMFPTrn,mfa,lbl.labeledpos,crig2);

%%
thisim = iPtLegsAllUsed';
idxD = thisim(:)';
idxD = [idxD idxD+19 idxD+2*19];
%% 
%pGT3dLegs = td.pGT(:,idxD);
pGT3dLegs = pGT3d(:,idxD);
pGT3dLegs3 = reshape(pGT3dLegs,[size(pGT3dLegs,1) 18 3]);

%% TrnDataSel
tblP = table(pGT3dLegs,'VariableNames',{'p'});
[grps,ffd,ffdiTrl] = CPRData.ffTrnSet(tblP,[]);
hFig1 = CPRData.ffTrnSetSelect(tblP,grps,ffd,ffdiTrl);
% Grand total of 481/515 (93%) shapes selected for training.
%%
iTrlSel = ffdiTrl{1}(1:481);
tblPSel = table(pGT3dLegs(iTrlSel,:),'VariableNames',{'p'});
[grps,ffd,ffdiTrl] = CPRData.ffTrnSet(tblPSel,[]);
hFig1 = CPRData.ffTrnSetSelect(tblPSel,grps,ffd,ffdiTrl);

%%
tMFPTrnSel = tMFPTrn(iTrlSel,:);
[I,pGT3d,bboxes,pGt3dRCerr,tMFPout] = rfCompileData3D(...
  tMFPTrnSel,mfa,lbl.labeledpos,crig2);
thisim = iPtLegsAllUsed';
idxD = thisim(:)';
idxD = [idxD idxD+19 idxD+2*19];
pGT3dLegs = pGT3d(:,idxD);
pGT3dLegs3 = reshape(pGT3dLegs,[size(pGT3dLegs,1) 18 3]);

%% check all legs have a coord
nrows = size(I,1);
pGt3d3 = reshape(pGT3d,nrows,19,3);
for iLeg=iLegsUse
  ipts = iPtLegs(iLeg,:);
  z = pGt3d3(:,ipts,:); 
  szassert(z,[nrows numel(ipts) 3]);
  assert(nnz(isnan(z))==0);
end
  
%% viz pGT with bboxes
figure('windowstyle','docked');
ax = axes;
hold(ax,'on');
MARKERS = {'o' 's' 'v'};
COLORS = {'b' 'g' 'r' 'k' 'y' 'm'};
for i=1:6
  iLeg = iLegsUse(i);
  iptsleg = iPtLegs(iLeg,:);
  for j=1:3
    ipt = iptsleg(j);
    xyz = squeeze(pGt3d3(:,ipt,:));
    szassert(xyz,[nrows 3]);
    scatter3(ax,xyz(:,1),xyz(:,2),xyz(:,3),20,COLORS{i},MARKERS{j},'filled');
  end
end
grid(ax,'on');
xlabel('x','fontweight','bold');
ylabel('y','fontweight','bold');
zlabel('z','fontweight','bold');
ax.XLim = [bboxes(1) bboxes(1)+bboxes(4)];
ax.YLim = [bboxes(2) bboxes(2)+bboxes(5)];
ax.ZLim = [bboxes(3) bboxes(3)+bboxes(6)];
%% Preprocessing
[I,pGT3d,bboxes] = rfCompileData3D(tMFPTrnSel,mfa,lbl.labeledpos,crig2);
%%
tblP = [tMFPTrnSel(:,{'movs' 'frm'}) table(pGT3dLegs,'VariableNames',{'p'})];
td = CPRData(I,tblP,repmat(bboxes,size(I,1),1));
%%
NTRIAL_SAMP = 30; % sample this many rows for intensity histogram
dTrl = floor(td.N/NTRIAL_SAMP);
I = td.I;
iTrls = 1:dTrl:size(I,1);

SIG1 = {[0 2 4 7] [0 2 4 7] [0 2 4 7 10]};
SIG2 = {[0 2 4 7] [0 2 4 7] [0 2 4 7 10]};
S = cell(1,3);
SGS = cell(1,3);
SLS = cell(1,3);
for iVw=1:3
  [S{iVw},SGS{iVw},SLS{iVw}] = Features.pp(I(iTrls,iVw),SIG1{iVw},SIG2{iVw},...
    'sRescale',false,'sgsRescale',false,'slsRescale',false);
end
%%
%[axS,axSGS,axSLS] = Features.ppViz(S,SGS,SLS,sig1,sig2,1);
%%
[S99p9,SGSmax,SGS99p9,SLSspn,SLSspn99,SLSspn98,SLSmu,SLSmdn] = ...
  cellfun(@(s,sgs,sls,sig1,sig2)Features.ppCalib(s,sgs,sls,sig1,sig2),...
    S,SGS,SLS,SIG1,SIG2,'uni',0);

%%
S99p9{:}
%%
SGS99p9{:}
%%
thisim = cellfun(@(x)cellfun(@diff,x),SLSspn99,'uni',0);
thisim{:}
%%
NAMES = {'rf left' 'rf right' 'rf bot'};
clear bpp;
bpp = cell(1,3);
ICHANS = {
  [2:4  ...
   ([2 3 6:11 13]+4) ... % SGS([2 3 6:11 13])
   ([2 3 5 7 9:12]+4+16)]; % SLS
  [2:4  ...
   ([2 3 6:11 13]+4) ... % SGS([2 3 6:11 13])
   ([2 3 5 7 9:12]+4+16)]; % SLS
  [(2:5) ... % all S except (1,1) which is same as orig image
   (2:3:23)+5 ... % SGS([2 5 8 ... 23])
   (4:3:25)+5+25] }; % SLS
  
for iVw=1:3
  bpp{iVw} = CPRBlurPreProc(NAMES{iVw},SIG1{iVw},SIG2{iVw});
  bpp{iVw}.iChan = ICHANS{iVw};
  
  bpp{iVw}.sRescale = true;
  bpp{iVw}.sRescaleFacs = 200./S99p9{iVw};
  bpp{iVw}.sgsRescale = true;
  bpp{iVw}.sgsRescaleFacs = 200./SGS99p9{iVw};
  bpp{iVw}.slsRescale = true;
  bpp{iVw}.slsRescaleFacs = 200./cellfun(@diff,SLSspn99{iVw});
end
save rfBlurPreProc bpp

%%
NAMES = {'rf left' 'rf right' 'rf bot'};
clear bpp;
bpp = cell(1,3);
SIG1 = {[0 2 4 8] [0 2 4 8] [0 2 4 8]};
SIG2 = {[0 2 4 8] [0 2 4 8] [0 2 4 8]};
ICHANS = { 2:4; 2:4; 2:4 };
  
for iVw=1:3
  bpp{iVw} = CPRBlurPreProc(NAMES{iVw},SIG1{iVw},SIG2{iVw});
  bpp{iVw}.iChan = ICHANS{iVw};
  
  bpp{iVw}.sRescale = false;
  bpp{iVw}.sgsRescale = false;
  bpp{iVw}.slsRescale = false;
end
save rfBPPSimple3D bpp

%% test calibs
rf = load('rfBlurPreProc.mat');
%%
td.computeIpp([],[],[],'romain',bpp,'iTrl',1:td.N);
%%
hFig = figure('windowstyle','docked');
axs = createsubplots(4,5);
iTrl = 342;
iVw = 1;
im = td.Ipp{iTrl,iVw};
info = td.IppInfo{iVw};
for i=1:20
  axes(axs(i));
  thisim = im(:,:,i);
  if true
    imhist(thisim);
    %hist(double(thisim(:)),255);
  else
    imagesc(thisim);
    colormap gray;
    caxis([0 255]);
  end
  fprintf('%s. 99ptile: %.3f. 99.9: %.3f\n',info{i},prctile(thisim(:),99),...
      prctile(thisim(:),99.9));
end

%%
GAMMA = .3;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);
arrayfun(@(x)colormap(x,mgray2),axS);

%% histeq I
% H0 = cell(1,NVIEW);
% tfuse = cell(1,NVIEW);
% for iView=1:NVIEW
%  [H0{iView},tfuse{iView}] = typicalImHist(I(:,iView));
% end
% 
% IHE = cell(size(I));
% for iRow=1:nrows
%   for iView=1:NVIEW
%     IHE{iRow,iView} = histeq(I{iRow,iView},H0{iView});
%   end
%   if mod(iRow,10)==0
%     fprintf(1,'histeq row %d\n',iRow);
%   end
% end
%% histeq
% Nlook = 5;
% axs = createsubplots(3,5);
% axs = reshape(axs,3,5);
% randrows = randint2(1,Nlook,[1 nrows]);
% GAMMA = .3;
% mgray = gray(256);
% mgray2 = imadjust(mgray,[],[],GAMMA);
% for iLook=1:Nlook
%   row = randrows(iLook);
%   for iVw=1:3
%     ax = axs(iVw,iLook);
%     axes(ax);
%     imagesc([I{row,iVw};IHE{row,iVw}]);
%     colormap(ax,mgray2);
%     axis(ax,'equal');
%   end
% end  

%% montage I/pGT
Imontage = I;
Nmont = 3;
figure('windowstyle','docked');
axs = createsubplots(Nmont,3);
axs = reshape(axs,Nmont,3);
randrows = randint2(1,Nmont,[1 nrows]);
GAMMA = .3;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);
for iMont=1:Nmont
  iRow = randrows(iMont);
  for iView=1:3
    ax = axs(iMont,iView);
    axes(ax);
    imagesc(Imontage{iRow,iView});
    colormap(ax,mgray2);
    axis(ax,'equal')
    
    if iView==1
      title(ax,num2str(iRow),'fontweight','bold');
    end
    if ~(iView==3 && iMont==1)
      ax.XTick = [];
      ax.YTick = [];
    end
  end
  
  X = cell(1,3);
  X{1} = squeeze(pGT3dLegs3(iRow,:,:))';
  %X{1} = squeeze(pGt3d3(iRow,:,:))';
  szassert(X{1},[3 18]);
  %szassert(X{1},[3 19]);
  X{2} = crig2.camxform(X{1},'lr');
  X{3} = crig2.camxform(X{1},'lb');
  
  MARKERS = {'o' 's' 'v'};
  COLORS = {[1 0 0] [0 1 0] [0 1 1] [1 1 0] [0 0 1] [1 0 1]};
  for i=1:6
    iLeg = iLegsUse(i);
    iptsleg = iPtLegs(iLeg,:);
    for j=1:3
      %ipt = iptsleg(j);
      ipt = (i-1)*3+j;
      for iView=1:3
        ax = axs(iMont,iView);
        hold(ax,'on');
        Xtmp = X{iView}(:,ipt);
        [r,c] = crig2.projectCPR(Xtmp,iView);
        plot(ax,c,r,[MARKERS{j}],'markersize',8,'color',COLORS{i},'markerfacecolor',COLORS{i});
      end
    end
  end
end



%%
PARAMFILE = 'f:\romain\tp@18pts@3d.yaml'; % KB: this is in .../forKB
sPrm = yaml.ReadYaml(PARAMFILE);
sPrm.Model.nviews = 3;
sPrm.Model.Prm3D.iViewBase = 1;
sPrm.Model.Prm3D.calrig = crig2;
rc = RegressorCascade(sPrm);
rc.init();


%%
% N = td.N;
% [Is,nChan] = td.getCombinedIs(1:td.N);
% bboxes = td.bboxes;
%%
nTrn = size(I,1);
pAll = rc.trainWithRandInit(I,repmat(bboxes,nTrn,1),pGT3dLegs);
pAll = reshape(pAll,nTrn,50,18*3,rc.nMajor+1);

%% Browse propagated replicates
TESTROWIDX = 222; % Pick any training row to view convergence
NPTS = 18;
frame = tMFPTrnSel.frm(TESTROWIDX);
if exist('lObj','var')==0
  lObj = Labeler;
  lObj.projLoadGUI(LBL); % KB: this won't be able to find movies will prompt you to locate
end
lObj.setFrameGUI(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
hLine = gobjects(3,NPTS);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 0 0;1 0 0; ...
          1 1 0;1 1 0;1 1 0; ...
          0 1 0;0 1 0;0 1 0; ...
          0 1 1;0 1 1;0 1 1; ...
          0 0 1;0 0 1;0 0 1; ...
          1 0 1;1 0 1;1 0 1];
          
  for iPt = 1:NPTS
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(pAll(TESTROWIDX,:,:,:));
szassert(pRepTrow,[50 18*3 rc.nMajor+1]);

for t=1:rc.nMajor+1
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,18,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:18 
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% Propagate on labeled, nontraining data
frmTest = 1:1000;
%frmTest = 10850:11849;
hWB = waitbar(0);
[ITest,tblTest] = Labeler.lblCompileContentsRaw(...
  lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,1,{frmTest},...
  'hWaitBar',hWB);
delete(hWB);

% NOTE: tblTest.p is projected/concatenated
%%
nTest = size(ITest,1);
[pAllTest,pIidxTest] = rc.propagateRandInit(ITest,repmat(bboxes,nTest,1),sPrm.TestInit);

%% Prune Propagated Replicates
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
nTestAug = sPrm.TestInit.Nrep;
pTstT = reshape(pAllTest,[nTest nTestAug trkD Tp1]);
%pTstT = pAllTest;

pTstTRed = nan(nTest,trkD,Tp1);
assert(sPrm.Prune.prune==1);
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,sPrm.Model,sPrm.Prune);
end
pTstTRedFinalT = pTstTRed(:,:,end);

%% Browse test frames in Labeler
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end

hLine = gobjects(3,NPTS);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 0 0;1 0 0; ...
          1 1 0;1 1 0;1 1 0; ...
          0 1 0;0 1 0;0 1 0; ...
          0 1 1;0 1 1;0 1 1; ...
          1 0 1;1 0 1;1 0 1; ...
          0 0 1;0 0 1;0 0 1];
  for iPt = 1:NPTS
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

nTest = size(pTstTRedFinalT,1);
pTstTRedFinalT = reshape(pTstTRedFinalT,nTest,18,3);
for iF=1:numel(frmTest)
  f = frmTest(iF);
  lObj.setFrameGUI(f);

  pTstBest = squeeze(pTstTRedFinalT(iF,:,:));
  for iVw=1:3
    for iPt=1:18
      X = pTstBest(iPt,:);
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  drawnow
  %input(sprintf('frame=%d',f));
end

%%
close all

%% Make a movie
OUTMOV = '3dmoo.avi';
FRAMERATE = 24;
GAMMA = .3;
mgray = gray(256);
mgray2 = imadjust(mgray,[],[],GAMMA);

bigim = nan(640+624,672);
mrs = lObj.movieReader;
hFig = figure;
ax = axes;
hIm = imagesc(bigim,'parent',ax);
colormap(ax,mgray2);
truesize(hFig);
hold(ax,'on');
ax.XTick = [];
ax.YTick = [];

hLine = gobjects(3,NPTS);
for iVw = 1:3
  clrs = [1 0 0;1 0 0;1 0 0; ...
          1 1 0;1 1 0;1 1 0; ...
          0 1 0;0 1 0;0 1 0; ...
          0 1 1;0 1 1;0 1 1; ...
          1 0 1;1 0 1;1 0 1; ...
          0 0 1;0 0 1;0 0 1];
  for iPt = 1:NPTS
    hLine(iVw,iPt) = plot(ax,nan,nan,'.',...
      'markersize',28,...
      'Color',clrs(iPt,:));
  end
end

vr = VideoWriter(OUTMOV);      
vr.FrameRate = FRAMERATE;
vr.open();

hTxt = text(10,15,'','parent',ax,'Color','white','fontsize',24);
hWB = waitbar(0,'Writing video');

for iF=1:numel(frmTest)
  f = frmTest(iF);
  
  imL = mrs(1).readframe(f);
  imR = mrs(2).readframe(f);
  imB = mrs(3).readframe(f);  
  bigim(24+(1:592),48+(1:288)) = imL;
  bigim(1:640,48+288+(1:288)) = imR;
  bigim(640+(1:624),1:672) = imB;  
  hIm.CData = bigim;
  
  pTstBest = squeeze(pTstTRedFinalT(iF,:,:));
  for iVw=1:3
    for iPt=1:18
      X = pTstBest(iPt,:);
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);

      switch iVw
        case 1
          rr = r+24;
          cc = c+48;
        case 2
          rr = r;
          cc = c+48+288;
        case 3
          rr = r+640;
          cc = c;
      end
      
      hL = hLine(iVw,iPt);
      set(hL,'XData',cc,'YData',rr);
    end
  end
  
  hTxt.String = sprintf('%04d',f);
  drawnow
  
  tmpFrame = getframe(ax);
  vr.writeVideo(tmpFrame);
  waitbar(iF/numel(frmTest),hWB,sprintf('Wrote frame %d\n',f));
end
       
vr.close();
delete(hTxt);
delete(hWB);

%% compare movs
vr1 = VideoReader('3d_nopp_allTrnData_20161115.avi');
vr2 = VideoReader('3d_nopp_481TrnData_20161116.avi');
%%
figure;
ax = createsubplots(1,2);
hIm1 = imagesc(uint8(zeros(1265,674,3)),'parent',ax(1));
hIm2 = imagesc(uint8(zeros(1265,674,3)),'parent',ax(2));
%%
f = 1;
while 1
  im1 = vr1.readFrame();
  im2 = vr2.readFrame();
  hIm1.CData = im1;
  hIm2.CData = im2;
  input(num2str(f));
end
  

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% END
%%%%%%%%%%%%%%%%%%%%


%%%%%%%
% OLD STUFF STARTING HERE
%%%%%%%%%%%%%


%% viz pGT in 3d
pGT2 = reshape(pGT_1_7_13,nRows,3,3);
clrs = parula(3);
hFig = figure('windowstyle','docked');
ax = axes;
hold(ax,'on');
for iRow = 1:nRows
  for iPt=1:3
    x = pGT2(iRow,iPt,1);
    y = pGT2(iRow,iPt,2);
    z = pGT2(iRow,iPt,3);
    plot3(ax,x,y,z,'o','MarkerFaceColor',clrs(iPt,:));
    text(x,y,z,num2str(iPt),'parent',ax,'Color',[0 0 0],'fontsize',12);
  end
end
ax.XLim = [bboxes(1) bboxes(1)+bboxes(4)];
ax.YLim = [bboxes(2) bboxes(2)+bboxes(5)];
ax.ZLim = [bboxes(3) bboxes(3)+bboxes(6)];



%% Browse original/recon labels for given frame/pt
% CONC: first one is mislabel in side view. second one is mislabel in
% bottom view. 
% IDEA: just recon from two pts that lead to most consistency?
%tRow = [1711 2032]; bad errReconR
tRow = [120 1711 1889 2289];
TROWIDX = 3;
frame = t19expandedLRB.frm(tRow);
lObj.setFrameGUI(frame(TROWIDX));
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
X = cell(1,3);
Xlr = cell(1,3);
X{1} = t19expandedLRB.XL(tRow(TROWIDX),:)';
X{2} = crig2.camxform(X{1},'lr');
X{3} = crig2.camxform(X{1},'lb');
Xlr{1} = t19expandedLRB.XLlr(tRow(TROWIDX),:)';
Xlr{2} = crig2.camxform(Xlr{1},'lr');
Xlr{3} = crig2.camxform(Xlr{1},'lb');
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  [r,c] = crig2.projectCPR(Xlr{iAx},iAx);
  hLine(iAx) = plot(ax,c,r,'wx','markersize',8);
end

%%
hold(ax,'on');
clrs = parula(19);
for iF=1:numel(frms)
  lposF = squeeze(lpos(:,3,:,frms(iF)));
  for iPt=1:19
    plot(ax,lposF(iPt,1),lposF(iPt,2),'o','Color',clrs(iPt,:));
  end
  input(num2str(iF));
end


%% pGT for 1-7-13
iPts_1_7_13 = [1 7 13];
pGT_1_7_13 = pGT(:,[iPts_1_7_13 iPts_1_7_13+19 iPts_1_7_13+38]);


%%
PARAMFILE = 'f:\romain\tp@3pts.yaml';
sPrm = yaml.ReadYaml(PARAMFILE);
sPrm.Model.nviews = 3;
sPrm.Model.Prm3D.iViewBase = 1;
sPrm.Model.Prm3D.calrig = crig2;
rc = RegressorCascade(sPrm);
rc.init();

%%
N = size(I,1);
pAll = rc.trainWithRandInit(I,repmat(bboxes,N,1),pGT_1_7_13);
pAll = reshape(pAll,197,50,9,31);

%% Browse trained replicates
pIidx = repmat((1:197)',50,1); % labels rows of pAll; indices into rows of tbl, I
TROWIDX = 1;
frame = tbl.frm(TROWIDX);
lObj.setFrameGUI(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
deleteValidGraphicsHandles(hLine);
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(pAll(TROWIDX,:,:,:));
szassert(pRepTrow,[50 9 31]);

for t=1:31
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,3,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:3
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% Propagate on labeled, nontraining data

% find all labeled frames not in tbl
frmTest = setdiff(tFrmPts.frm,tbl.frm);

[ITest,tblTest] = Labeler.lblCompileContentsRaw(...
  lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,1,{frmTest},...
  'hWB',waitbar(0));
% NOTE: tblTest.p is projected/concatenated
%%
nTest = size(ITest,1);
[p_t,pIidx] = rc.propagateRandInit(ITest,repmat(bboxes,nTest,1),sPrm.TestInit);
p_t = reshape(p_t,nTest,50,9,31);

%% Browse propagated replicates
TESTROWIDX = 155;
frame = tblTest.frm(TESTROWIDX);
lObj.setFrameGUI(frame);
%lposCurr = squeeze(lpos(4,:,:,11952)); % 3x2
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',20,...
      'Color',clrs(iPt,:));
  end
end

pRepTrow = squeeze(p_t(TESTROWIDX,:,:,:));
szassert(pRepTrow,[50 9 31]);

for t=1:31
  pRep = pRepTrow(:,:,t);
  pRep = reshape(pRep,50,3,3); % (iRep,iPt,iDim)
  for iVw=1:3
    for iPt=1:3
      X = squeeze(pRep(:,iPt,:)); % [50x3]
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('t=%d',t));
end

%% PRUNE PROPAGATED REPLICATES
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
nTestAug = sPrm.TestInit.Nrep;
pTstT = reshape(p_t,[nTest nTestAug trkD Tp1]);

pTstTRed = nan(nTest,trkD,Tp1);
assert(sPrm.Prune.prune==1);
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,sPrm.Model,sPrm.Prune);
end
pTstTRedFinalT = pTstTRed(:,:,end);

%% Browse test frames
axAll = lObj.gdata.axes_all;
if exist('hLine','var')>0
  deleteValidGraphicsHandles(hLine);
end
hLine = gobjects(3,3);
for iAx = 1:3
  ax = axAll(iAx);
  hold(ax,'on');
  clrs = [1 0 0;1 1 0;0 1 0];
  for iPt = 1:3
    hLine(iAx,iPt) = plot(ax,nan,nan,'.',...
      'markersize',30,...
      'Color',clrs(iPt,:));
  end
end

pTstTRedFinalT2 = reshape(pTstTRedFinalT2,nTest2,3,3);
for iF=1:numel(frmTest2)
  f = frmTest2(iF);
  lObj.setFrameGUI(f);

  pTstBest = squeeze(pTstTRedFinalT2(iF,:,:));
  for iVw=1:3
    for iPt=1:3
      X = pTstBest(iPt,:);
      Xvw = crig2.viewXformCPR(X',1,iVw); % iViewBase==1
      [r,c] = crig2.projectCPR(Xvw,iVw);
      
      h = hLine(iVw,iPt);
      set(h,'XData',c,'YData',r);
    end
  end
  
  input(sprintf('frame=%d',f));
end

%% Propagate on ENTIRE MOVIE ish

frmTest2 = 1000:5:3500;
[ITest2,tblTest2] = Labeler.lblCompileContentsRaw(...
  lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,1,{frmTest2},...
  'hWaitBar',waitbar(0));
% NOTE: tblTest.p is projected/concatenated

%%
nTest2 = size(ITest2,1);
[p_t2,pIidx2] = rc.propagateRandInit(ITest2,repmat(bboxes,nTest2,1),sPrm.TestInit);
p_t2 = reshape(p_t2,nTest2,50,9,31);

%%
trkD = rc.prmModel.D;
Tp1 = rc.nMajor+1;
nTestAug = sPrm.TestInit.Nrep;
pTstT2 = reshape(p_t2,[nTest2 nTestAug trkD Tp1]);

pTstTRed2 = nan(nTest2,trkD,Tp1);
assert(sPrm.Prune.prune==1);
for t = 1:Tp1
  fprintf('Pruning t=%d\n',t);
  pTmp = permute(pTstT2(:,:,:,t),[1 3 2]); % [NxDxR]
  pTstTRed2(:,:,t) = rcprTestSelectOutput(pTmp,sPrm.Model,sPrm.Prune);
end
pTstTRedFinalT2 = pTstTRed2(:,:,end);