function tblMF = labelAddLabelsMFTableStc(tblMF,lbls,varargin)
  % Add label/trx information to an MFTable
  %
  % tblMF (output): Same rows as tblMF, but with addnl label-related
  %   fields as in labelGetMFTableLabeledStc
  %
  % tblMF: MFTable with flds MFTable.FLDSID. tblMF.mov are 
  %   MovieIndices. tblMF.mov.get() are indices into lbls; ie lbls must
  %   be the appropriate cell-arr-of-labels to be indexed by tblMF.mov
  %
  % lbls: cell array of Labels structs. 
  
  [trxFilesAllFull,trxCache,wbObj,isma,rois,maxanimals] = myparse(varargin,...
    'trxFilesAllFull',[],... % cellstr, indexed by tblMV.mov. if supplied, tblMF will contain .pTrx field
    'trxCache',[],... % must be supplied if trxFilesAllFull is supplied
    'wbObj',[],... % optional WaitBarWithCancel or ProgressMeter. If canceled, tblMF (output) indeterminate
    'isma',false, ...
    'roi',[],...
    'maxanimals',1 ...
    );      
  tfWB = ~isempty(wbObj);
  
  assert(istable(tblMF));
  tblfldscontainsassert(tblMF,MFTable.FLDSID);
  nMov = numel(lbls);

  
  tfTrx = ~isempty(trxFilesAllFull);
  if tfTrx
    nView = size(trxFilesAllFull,2);
    szassert(trxFilesAllFull,[nMov nView]);
    tfTfafEmpty = cellfun(@isempty,trxFilesAllFull);
    % Currently, projects allowed to have some movs with trxfiles and
    % some without.
    assert(all( all(tfTfafEmpty,2) | all(~tfTfafEmpty,2) ),...
      'Unexpected trxFilesAllFull specification.');
    tfMovHasTrx = all(~tfTfafEmpty,2); % tfTfafMovEmpty(i) indicates whether movie i has trxfiles        
  else
    nView = 1;
  end

  nrow = height(tblMF);
  
  if tfWB && nrow>0 ,
    if isa(wbObj, 'ProgressMeter') ,
      wbObj.start('message', 'Compiling labels', ...
                  'denominator',nrow) ;
      oc = onCleanup(@()(wbObj.finish())) ;          
    else
      wbObj.startPeriod('Compiling labels','shownumden',true,...
                        'denominator',nrow);
      oc = onCleanup(@()wbObj.endPeriod);
    end
    wbtime = tic;
    maxwbtime = .1; % update waitbar every second
  end
  
  % Could also leverage Labels.totable and then do joins.
  
  npts = lbls{1}.npts;
  if isma
    pAcc = nan(nrow,maxanimals,npts*2);
    pTSAcc = -inf(nrow,maxanimals,npts);
    tfoccAcc = false(nrow,maxanimals,npts);
    roi = nan(nrow,maxanimals,4,2); % assume max number of rois is less than max number of animals
  else
    pAcc = nan(nrow,npts*2);
    pTSAcc = -inf(nrow,npts);
    tfoccAcc = false(nrow,npts);
    % roi = nan(nrow,4,2);
  end
  pTrxAcc = nan(nrow,nView*2); % xv1 xv2 ... xvk yv1 yv2 ... yvk
  thetaTrxAcc = nan(nrow,nView);
  aTrxAcc = nan(nrow,nView);
  bTrxAcc = nan(nrow,nView);
  tfInvalid = false(nrow,1); % flags for invalid rows of tblMF encountered
  
  iMovsAll = tblMF.mov.get;
  frmsAll = tblMF.frm;
  iTgtAll = tblMF.iTgt;
  
  iMovsUnique = unique(iMovsAll);
  nRowsComplete = 0;
  
  for movIdx = 1:numel(iMovsUnique),
    iMov = iMovsUnique(movIdx);
    rowsCurr = find(iMovsAll == iMov); % absolute row indices into tblMF

    s = lbls{iMov};
    if isempty(rois)
      r = LabelROI.new();
    else
      r = rois{iMov};
    end
           
    if tfTrx && tfMovHasTrx(iMov)
      NFRMS = [];
      % By passing [], either trxCache already contains this trxfile
      % and nfrms has been recorded for that mov/trx, or the maximum
      % trx.endFrame will be used.
      [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
        trxCache,trxFilesAllFull(iMov,:),NFRMS);
      
      assert(isscalar(trxI),'Multiview projs with trx currently unsupported.');
      trxI = trxI{1};
    end
    
    for jrow = 1:numel(rowsCurr),
      irow = rowsCurr(jrow); % absolute row index into tblMF
      
      if tfWB && nrow>0 && toc(wbtime) >= maxwbtime,
        wbtime = tic() ;
        if isa(wbObj, 'ProgressMeter') ,
          wbObj.bump(nRowsComplete) ;
          if wbObj.wasCanceled ,
            return
          end
        else
          tfCancel = wbObj.updateFracWithNumDen(nRowsComplete);
          if tfCancel
            return
          end
        end
      end
      
      %tblrow = tblMF(irow,:);
      frm = frmsAll(irow);
      iTgt = iTgtAll(irow);
      
      if frm<1
        tfInvalid(irow) = true;
        continue;
      end
      
      if tfTrx && tfMovHasTrx(iMov)
        % will harderr if frm is out of bounds of frm2trxtotAnd
        tgtLiveInFrm = frm2trxTotAnd(frm,iTgt); 
        if ~tgtLiveInFrm
          tfInvalid(irow) = true;
          continue;
        end
      else
        %assert(iTgt==1);
      end
      
      if isma
        [~,p,occ,ts] = Labels.isLabeledFMA(s,frm);
        % p and occ have appropriate size/vals even if tf 
        % (first out arg) is false
        nl = size(p,2);
        pAcc(irow,1:nl,:) = p';
        pTSAcc(irow,1:nl,:) = ts';
        tfoccAcc(irow,1:nl,:) = occ'; 
        roif = (r.f==frm);
        nroi = nnz(roif);
        roi(irow,1:nroi,:,:) = permute(r.verts(:,:,roif),[3,1,2]);
      else
        [~,p,occ,ts] = Labels.isLabeledFT(s,frm,iTgt);
        % p and occ have appropriate size/vals even if tf 
        % (first out arg) is false
        pAcc(irow,:) = p;
        pTSAcc(irow,:) = ts;
        tfoccAcc(irow,:) = occ; 
      end
      
      if tfTrx && tfMovHasTrx(iMov)
        %xtrxs = cellfun(@(xx)xx(iTgt).x(frm+xx(iTgt).off),trxI);
        %ytrxs = cellfun(@(xx)xx(iTgt).y(frm+xx(iTgt).off),trxI);
        trxItgt = trxI(iTgt);
        frmabs = frm + trxItgt.off;
        xtrxs = trxItgt.x(frmabs);
        ytrxs = trxItgt.y(frmabs);
        
        pTrxAcc(irow,:) = [xtrxs(:)' ytrxs(:)']; 
        %thetas = cellfun(@(xx)xx(iTgt).theta(frm+xx(iTgt).off),trxI);
        thetas = trxItgt.theta(frmabs);
        thetaTrxAcc(irow,:) = thetas(:)'; 
        
%             as = cellfun(@(xx)xx(iTgt).a(frm+xx(iTgt).off),trxI);
%             bs = cellfun(@(xx)xx(iTgt).b(frm+xx(iTgt).off),trxI);
        as = trxItgt.a(frmabs);
        bs = trxItgt.b(frmabs);            
        aTrxAcc(irow,:) = as(:)'; 
        bTrxAcc(irow,:) = bs(:)'; 
      else
        % none; these arrays pre-initted to nan
        
%             pTrxAcc(irow,:) = nan; % singleton exp
%             thetaTrxAcc(irow,:) = nan; % singleton exp
%             aTrxAcc(irow,:) = nan; 
%             bTrxAcc(irow,:) = nan; 
      end
      nRowsComplete = nRowsComplete + 1;
    end
  end
  
  if isma
    tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,roi,...
      'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx','roi'});
  else
    tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,...
      'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx'});
  end
  tblMF = [tblMF tLbl];
  
  if any(tfInvalid)
    warningNoTrace('Removed %d invalid rows of MFTable.',nnz(tfInvalid));
    tblMF = tblMF(~tfInvalid,:);
  end      
end  % function
