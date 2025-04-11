%%
t = readtable('multitarget_bubble_expandedbehavior_20180107_notes_closelabeledframes_trackingmaskedorunmasked.csv');
tMFT = t(:,1:3);

%%
masknotes = t.maskingNotes;
tf = cellfun(@numel,masknotes)>1;
masknotes(tf) = {'n'};
masknotes = lower(masknotes);
masknotes = categorical(masknotes);

%%
FGTHRESH = 6;
ROWSTOPROC = find(masknotes=='n'); % either didn't mask, or didn't mask closest etc
nproc = numel(ROWSTOPROC);

hFig = figure;
hTG = uitabgroup('Parent',hFig);
  
for iproc=1:nproc
  IROW = ROWSTOPROC(iproc);
  hT = uitab(hTG,'Title',sprintf('Row%02d',IROW));
  
  trow = tMFT(IROW,:);
  iMov = trow.mov;
  if iMov~=lObj.currMovie
    lObj.movieSetGUI(iMov);
    pause(10);
  end

  mr = lObj.movieReader;
  im = mr.readframe(trow.frm);
  im = double(im);

  movfile = lObj.movieFilesAllFull{iMov,1};
  movinfo = lObj.movieInfoAll{iMov};
  [imbg,imbgdev,n_bg_std_thresh_low] = flyBubbleBGRead(movfile,movinfo.info);
  assert(n_bg_std_thresh_low==115);

  trxfile = lObj.trxFilesAllFull{iMov};
  [trx,frm2trx] = lObj.getTrx(trxfile,movinfo.nframes);

  pdfLeg = load('f:\aptMultiTargetAssignPx20171016\pdfLeg20171024.mat');
  pdfImAlgn = load('f:\aptNborMaskDeux20180215\pdfImAlgn20180218.mat');
  [imccGMM,imccGMMpre] = PxAssign.asgnGMMglobal(im,imbg,imbgdev,trx,trow.frm,'fgthresh',FGTHRESH);
  imccCC = PxAssign.asgnCC(im,imbg,imbgdev,trx,trow.frm,'fgthresh',FGTHRESH);
  [imccPDF,imccPDFpre,pdfTgts] = PxAssign.asgnPDF(im,imbg,imbgdev,trx,...
    trow.frm,pdfLeg.plegnormS,pdfLeg.xe,pdfLeg.ye,pdfLeg.amu,pdfLeg.bmu,'fgthresh',FGTHRESH);
  [imccPDFimf,imccPDFpreimf,pdfTgtsimf] = PxAssign.asgnPDF(im,imbg,imbgdev,trx,...
    trow.frm,pdfImAlgn.imforeBWAlgnNorm,pdfImAlgn.xroiedge,pdfImAlgn.yroiedge,...
    pdfImAlgn.amu,pdfImAlgn.bmu,'fgthresh',FGTHRESH);
  
  [imtgtGMM,imnottgtGMM] = PxAssign.performMask(im,imbg,imccGMM,trx,...
    trow.tgt,trow.frm);
  [imtgtCC,imnottgtCC] = PxAssign.performMask(im,imbg,imccCC,trx,...
    trow.tgt,trow.frm);
  [imtgtPDF,imnottgtPDF] = PxAssign.performMask(im,imbg,imccPDF,trx,...
    trow.tgt,trow.frm);
  [imtgtPDFimf,imnottgtPDFimf] = PxAssign.performMask(im,imbg,imccPDFimf,trx,...
    trow.tgt,trow.frm);


  %%
  ROIRAD = 90;
  [xctr,yctr] = PxAssign.trxCtrRound(trx(trow.tgt),trow.frm);
  [xlo,xhi,ylo,yhi] = PxAssign.roiTrxCtr(xctr,yctr,ROIRAD);
  roiFcn = @(x)x(ylo:yhi,xlo:xhi);
  imroi = roiFcn(im);

  imccCCroi = roiFcn(imccCC);
  imtgtCCroi = roiFcn(imtgtCC);
  imnottgtCCroi = roiFcn(imnottgtCC);

  imccGMMroi = roiFcn(imccGMM);
  imtgtGMMroi = roiFcn(imtgtGMM);
  imnottgtGMMroi = roiFcn(imnottgtGMM);
  imccGMMpreroi = roiFcn(imccGMMpre);

  imccPDFroi = roiFcn(imccPDF);
  imtgtPDFroi = roiFcn(imtgtPDF);
  imnottgtPDFroi = roiFcn(imnottgtPDF);
  imccPDFpreroi = roiFcn(imccPDFpre);

  imccPDFimfroi = roiFcn(imccPDFimf);
  imtgtPDFimfroi = roiFcn(imtgtPDFimf);
  imnottgtPDFimfroi = roiFcn(imnottgtPDFimf);
  imccPDFpreimfroi = roiFcn(imccPDFpreimf);

  %% track the masked images
  trowTmp = trow;
  trowTmp.Properties.VariableNames{3} = 'iTgt';
  
  [~,~,~,oTheta] = PxAssign.getTrxStuffAtFrm(trx(trow.tgt),trow.frm);
  rc = lObj.tracker.trnResRC;
  sPrm = lObj.tracker.sPrm;

  [imnrroi,imncroi] = size(imroi);
  bbox = [1 1 imncroi imnrroi];
  trkD = sPrm.Model.nfids*2;
  Tp1 = rc.nMajor+1;

  p_t = rc.propagateRandInit({uint8(imtgtCCroi)},bbox,sPrm.TestInit,'orientationThetas',oTheta);  
  pTst = reshape(p_t,[1 sPrm.TestInit.Nrep trkD Tp1]);
  pTstCC = CPRLabelTracker.applyPruning(pTst(:,:,:,end),trowTmp,sPrm.Prune);
  szassert(pTstCC,[1 trkD]);

  p_t = rc.propagateRandInit({uint8(imtgtGMMroi)},bbox,sPrm.TestInit,'orientationThetas',oTheta);  
  pTst = reshape(p_t,[1 sPrm.TestInit.Nrep trkD Tp1]);
  pTstGMM = CPRLabelTracker.applyPruning(pTst(:,:,:,end),trowTmp,sPrm.Prune);
  szassert(pTstGMM,[1 trkD]);

  p_t = rc.propagateRandInit({uint8(imtgtPDFroi)},bbox,sPrm.TestInit,'orientationThetas',oTheta);  
  pTst = reshape(p_t,[1 sPrm.TestInit.Nrep trkD Tp1]);
  pTstPDF = CPRLabelTracker.applyPruning(pTst(:,:,:,end),trowTmp,sPrm.Prune);
  szassert(pTstPDF,[1 trkD]);
  
  p_t = rc.propagateRandInit({uint8(imtgtPDFimfroi)},bbox,sPrm.TestInit,'orientationThetas',oTheta);  
  pTst = reshape(p_t,[1 sPrm.TestInit.Nrep trkD Tp1]);
  pTstPDFimf = CPRLabelTracker.applyPruning(pTst(:,:,:,end),trowTmp,sPrm.Prune);
  szassert(pTstPDFimf,[1 trkD]);
  
  %%
%   hFig = figure('windowstyle','docked','tag','imfigs','position',[1 1 1859 1430]);
  axs = mycreatesubplots(4,5,.01,hT);
  axs = reshape(axs,4,5);

  ax = axs(1,1);
  axes(ax);
  imagesc(imccCCroi);
  colormap(ax,'jet');
  axis image;
  tstr = sprintf('(%d,%d,%d)',trow.mov,trow.frm,trow.tgt);
  hTxt = text(5,10,tstr,'Color',[1 1 1],'fontsize',14,'fontweight','bold');

  ax = axs(1,2);
  axes(ax);
  imagesc(imtgtCCroi);
  colormap(ax,'gray');
  hold(ax,'on');
  plot(pTstCC(1:17),pTstCC(18:end),'.','markersize',10,'color',[1 0 0]);
  axis image;

  ax = axs(1,3);
  axes(ax);
  imagesc(imtgtCCroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(1,4);
  axes(ax);
  imagesc(imnottgtCCroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(2,1);
  axes(ax);
  imagesc(imccGMMroi);
  colormap(ax,'jet');
  axis image;
  
  ax = axs(2,2);
  axes(ax);
  imagesc(imtgtGMMroi);
  colormap(ax,'gray');
  hold(ax,'on');
  plot(pTstGMM(1:17),pTstGMM(18:end),'.','markersize',10,'color',[1 0 0]);
  axis image;

  ax = axs(2,3);
  axes(ax);
  imagesc(imtgtGMMroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(2,4);
  axes(ax);
  imagesc(imnottgtGMMroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(2,5);
  axes(ax);
  imagesc(imccGMMpreroi);
  colormap(ax,'jet');
  axis image;

  ax = axs(3,1);
  axes(ax);
  imagesc(imccPDFroi);
  colormap(ax,'jet');
  axis image;
  
  ax = axs(3,2);
  axes(ax);
  imagesc(imtgtPDFroi);
  colormap(ax,'gray');
  hold(ax,'on');
  plot(pTstPDF(1:17),pTstPDF(18:end),'.','markersize',10,'color',[1 0 0]);
  axis image;

  ax = axs(3,3);
  axes(ax);
  imagesc(imtgtPDFroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(3,4);
  axes(ax);
  imagesc(imnottgtPDFroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(3,5);
  axes(ax);
  imagesc(imccPDFpreroi);
  colormap(ax,'jet');
  axis image;
  
  ax = axs(4,1);
  axes(ax);
  imagesc(imccPDFimfroi);
  colormap(ax,'jet');
  axis image;

  ax = axs(4,2);
  axes(ax);
  imagesc(imtgtPDFimfroi);
  colormap(ax,'gray');
  hold(ax,'on');
  plot(pTstPDFimf(1:17),pTstPDFimf(18:end),'.','markersize',10,'color',[1 0 0]);
  axis image;

  ax = axs(4,3);
  axes(ax);
  imagesc(imtgtPDFimfroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(4,4);
  axes(ax);
  imagesc(imnottgtPDFimfroi);
  colormap(ax,'gray');
  axis image;

  ax = axs(4,5);
  axes(ax);
  imagesc(imccPDFpreimfroi);
  colormap(ax,'jet');
  axis image;
  
  for i=1:numel(axs)
    set(axs(i),'XTick',[],'YTick',[]);
  end
  
  linkaxes(axs);
end

%% 20170227 Compare production masking in APT on flybub ~3700 rows

% Procedure: take a proj, turn masking on, do a train just to see
% masked/preproced data in tracker.data.

% CONC: 
% - conncomp doesn't split touching flies (duh) so the number of fg pxs
% masked for conncomp tracks that for gmmem except when it is sometimes way
% lower for conncomp.
% - gmmem and emppdf track pretty closely. very rarely, there are 
% deviations. Looked at largest deviation: (mov 4, frm 20718, tgt 12). 
% gmmem does an odd grouping (eg two flies headbutting, one huge ellipse 
% containing both and one much smaller on one fly).

nm0 = load('nmaskMDnborMaskConnCompFG5.mat');
nm1 = load('nmaskMDnborMaskGMMEMFG5.mat');
nm2 = load('nmaskMDnborMaskEmpPDFFG5.mat');
nm0 = nm0.nmask;
nm1 = nm1.nmask;
nm2 = nm2.nmask;
nm = [nm0 nm1 nm2];

figure
scatter(nm0,nm1);
xlabel('conncomp','fontweight','bold');
ylabel('gmmem','fontweight','bold');
tstr = sprintf('Num fg px masked in %d flybub lbled rows',size(nm0,1));
title(tstr,'fontweight','bold');
grid on;

figure
scatter(nm1,nm2);
xlabel('gmmem','fontweight','bold');
ylabel('emppdf','fontweight','bold');
tstr = sprintf('Num fg px masked in %d flybub lbled rows',size(nm1,1));
title(tstr,'fontweight','bold');
grid on;