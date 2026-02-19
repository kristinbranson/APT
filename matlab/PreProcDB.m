classdef PreProcDB < handle
  % Preprocessed DB for DL
  %
  % We are forking this off the CPR/old .preProcData stuff in Labeler b/c:
  % 1) of rotated cache needs (potentially an imposition on CPR, and 
  % ideally we don't touch CPR for now)
  % 1a) existing diffs in preproc pipelines for CPR and DL will probably 
  % continue to exist into the medium term.
  % 2) cleanup/refactor wherein preProcData* Labeler meths don't have an
  % ideal API. Eventually CPR will utilize the/a PreProcDB.
  
  properties 
    dat % CPRData scalar
    tsLastEdit % last edit timestamp
  end
  
  methods
    
    function obj = PreProcDB()
    end
    
    function init(obj)
      I = cell(0,1);
      tblP = MFTable.emptyTable(MFTable.FLDSCORE);
      obj.dat = CPRData(I,tblP);
      obj.tsLastEdit = now;
    end
    
    % AL 20200804. Regarding cropping/rotating-to-up. PreProcDB currently
    % does these operations itself rather than relying on anything existing
    % (eg .p, .pRel) in incoming tables. Those fields are used by the CPR 
    % pipeline. The situation is complicated right now as CPR handles
    % pp.TargetCrop.AlignUsingTrxTheta differently than the DL pipeline.
    %
    %  Inputs/Outputs for cache
    %
    %  case: trx, TargetCrop.AlignUsingTrxTheta=true.
    %   it is assumed that trx are constant, ie (m,f,t) fixes trx.x,y,th. 
    %   .roi is used but only for the roiRad. (it is centered at the trx 
    %   but this is not the actual roi of the im). this is a bit silly but
    %   nbd.
    %   
    %   .mft, .pAbs, .tfocc, .roi (for roiRad), prmPP.TargetCrop
    %        => im-in-cache, p-in-cache
    % 
    %  case: trx, AlignUsingTrxTheta=false.
    %   here .roi is used for cropping, although the .p is again recomputed
    %   from .pAbs (which seems goodish). MD.roi is the actual roi of the 
    %   im.
    %    
    %  case: crop (no trx)
    %   should be like trx, AlignUsingTrxTheta=false case.
    %  
    %  case: no crop, no trx
    %
    %  So in general, it is 
    %   .mft, .pAbs, .tfocc, .roi, prmPP.TargetCrop => im-in-cache, p-in-cache
    %  * changing prmPP should clear the cache
    %  * changing crop defns should clear the cache

        
    function [tblNewReadFailed,dataNew,tfReadFailed] = ...
                                    add(obj,tblNew,lObj,varargin)
      % Add new rows to DB
      %
      % tblNew: new rows. MFTable.FLDSCORE are required fields. .roi may 
      %   be present and if so WILL BE USED to grab images and included in 
      %   data/MD. Other fields are ignored.
      %
      %   **VERY IMPORTANT**: This method uses tblNew.pAbs, and NEVER
      %   tblNew.p. tblNew.p has often been massaged to be relative to .roi
      %   (when .roi is present). This method does its own cropping to
      %   generate image patches and only it knows how .p must be massaged.
      %
      %   AL 20200804. We now store .pAbs in the .dat.MD for comparison to
      %   incoming rows.
      %
      % tblNewReadFailed: table of failed-to-read rows. Currently subset of
      %   tblNew. If non-empty, then .dat was not updated with these rows 
      %   as requested.
      % dataNew: CPRData of new data created/added
      
      [wbObj,prmsTgtCrop,computeOnly] = myparse(varargin,...
        'wbObj',[],...
        'prmsTgtCrop',[],... % preprocessing params
        'computeOnly',false... % if true, don't update the DB, compute only.
        );

      tfWB = ~isempty(wbObj);
      if isempty(prmsTgtCrop)
        prms = lObj.trackParams;
        if isempty(prms)
          error('Please specify parameters.');
        end
        prmsTgtCrop = prms.ROOT.MultiAnimal.TargetCrop;
      end      
%       assert(isstruct(prmpp),'Expected parameters to be struct/value class.');

      FLDSREQUIRED = [MFTable.FLDSCORE 'pAbs'];
      % AL 20200804. Store pAbs in MD. This is the original/raw lbl. We
      % store this so we can compare incoming rows to see if they need an
      % update in addAndUpdate().
      FLDSALLOWED = [MFTable.FLDSCORE {'roi' 'nNborMask' 'tformA' 'pAbs'}];
      tblfldscontainsassert(tblNew,FLDSREQUIRED);
      
      tblNew.p = tblNew.pAbs;
      
      currMD = obj.dat.MD;
      tf = tblismember(tblNew,currMD,MFTable.FLDSID);
      nexists = nnz(tf);
      if nexists>0
        error('%d rows of tblNew exist in current db.',nexists);
      end
      
      tblNewReadFailed = tblNew([],:);
      dataNew = [];
      
      PRMPP_DUMMY = struct;
      PRMPP_DUMMY.histeq = false;
      PRMPP_DUMMY.BackSub.Use = false;
      PRMPP_DUMMY.NeighborMask.Use = false;
                        
      tblNewConc = lObj.mftTableConcretizeMov(tblNew);
      nNew = height(tblNew);
      if nNew>0
        fprintf(1,'Adding %d new rows to data...\n',nNew);
        
        [I,nNborMask,didread,tformA] = CPRData.getFrames(tblNewConc,...
          'wbObj',wbObj,...
          'forceGrayscale',lObj.movieForceGrayscale,...
          'preload',lObj.movieReadPreLoadMovies,...
          'movieInvert',lObj.movieInvert,...          
          'roiPadVal',prmsTgtCrop.PadBkgd,...
          'rotateImsUp',prmsTgtCrop.AlignUsingTrxTheta&~isempty(lObj.trxCache),...
          'isDLpipeline',true,...
          'doBGsub',PRMPP_DUMMY.BackSub.Use,...
          'bgReadFcn',[],...
          'bgType',[],...
          'maskNeighbors',PRMPP_DUMMY.NeighborMask.Use,...
          'maskNeighborsMeth',[],...
          'maskNeighborsEmpPDF',[],...
          'fgThresh',[],...
          'trxCache',lObj.trxCache,...
          'labeler',lObj,...
          'usePreProcData',false);
        if tfWB && wbObj.isCancel
          % obj unchanged
          return;
        end
        % Include only FLDSALLOWED in metadata to keep CPRData md
        % consistent (so can be appended)
        
        didreadallviews = all(didread,2);
        tblNewReadFailed = tblNew(~didreadallviews,:);
        tblNew(~didreadallviews,:) = [];
        I(~didreadallviews,:) = [];
        nNborMask(~didreadallviews,:) = [];
        tformA(:,:,~didreadallviews,:) = [];
        tfReadFailed = ~didreadallviews;
        
        % AL: a little worried if all reads fail -- might get a harderr
        
        tfColsAllowed = ismember(tblNew.Properties.VariableNames,FLDSALLOWED);
        tblNewMD = tblNew(:,tfColsAllowed);
        tblNewMD = [tblNewMD table(nNborMask)];
        
        pAbs = tblNewMD.p; % atm tblNewMD.p matches tblNewMD.pAbs
        n = height(tblNewMD);
        nPhysPts = lObj.nPhysPoints;
        nView = lObj.nview;
        pRel = reshape(pAbs,[n nPhysPts nView 2]);
        %pRelnan = isnan(pRel);
        szassert(tformA,[3 3 n nView]);
        %tformTinvAll = nan(n,3*2*nView); % tform arrays
        for i=1:n
          for ivw=1:nView
            tform = maketform('affine',tformA(:,:,i,ivw));
            % Go through this rigamarole as tformfwd() errs when passed any 
            % nans
            u = pRel(i,:,ivw,1);
            v = pRel(i,:,ivw,2);
            tfnan = isnan(u) | isnan(v); % isnan(u) should prob EQUAL isnan(v)
            x = nan(1,nPhysPts);
            y = nan(1,nPhysPts);
            [x(~tfnan),y(~tfnan)] = tformfwd(tform,u(~tfnan),v(~tfnan));
            pRel(i,:,ivw,1) = x;
            pRel(i,:,ivw,2) = y;
%             [pRel(i,:,ivw,1),pRel(i,:,ivw,2)] = ...
%               tformfwd(tform,pRel(i,:,ivw,1),pRel(i,:,ivw,2));
%             tformTinv = tform.tdata.Tinv;
%             assert(isequal(tformTinv(:,3),[0;0;1]));
%             tformTinvAll(i,(1:6)+(ivw-1)*6) = tformTinv(1:6);
          end
        end
        pRel = reshape(pRel,[n nPhysPts*nView*2]);
        tblNewMD.p = pRel; % now we have tblNewMD.p (which is prel) and .pAbs
        tformA = permute(tformA,[3 1 2 4]); % [n 3 3 nView]
        assert(isequal(tformA(:,:,3,:),repmat([0 0 1],n,1,1,nView)));
        tformA = tformA(:,:,1:2,:); % [n 3 2 nView]
        tblNewMD.tformA = reshape(tformA,n,[]);
        
        dataNew = CPRData(I,tblNewMD);
        
        if ~computeOnly
          obj.dat.append(dataNew);
          obj.tsLastEdit = now;
        end
      end
    end
    
    function [tblAddReadFailed,tfAU,locAU] = ...
                        addAndUpdate(obj,tblAU,lObj,varargin)
      % update DB 
      %
      % tblAU: ("tblAddUpdate")
      %   - required fields: [MFTable.FLDSCORE 'pAbs']. Note .p is not
      %   used, .pAbs is. See comments above
      %   - .roi: optional, USED WHEN PRESENT. (prob needs to be either
      %   consistently there or not-there for a given obj or initData()
      %   "session"
      %
      % tblAddReadFailed: tbl of failed-read adds. will have height 0 if
      %   everything went well.
      % tfAU: [tfAU,locAU] = tblismember(tblAU,obj.dat.MD,MFTable.FLDSID)
      % locAU: 
      
      [wbObj,prmsTgtCrop,verbose] = myparse(varargin,...
        'wbObj',[], ... % WaitBarWithCancel. If cancel, obj unchanged.
        'prmsTgtCrop',[], ...
        'verbose',true ...
        );
      
      FLDSCMP = {'mov' 'frm' 'iTgt' 'tfocc' 'pAbs'};
      
      tblMD = obj.dat.MD;
      
      % similar to MFTable.tblPDiff
      [tfexistAU,locexist] = tblismember(tblAU,tblMD,MFTable.FLDSID);
      if height(tblMD)>0
        % special-case bc tblMD will not have right flds when empty
        
        [tfsameAU,locsame] = tblismember(tblAU,tblMD,FLDSCMP);
        assert(isequal(locsame(tfsameAU),locexist(tfsameAU)));
        if nnz(tfsameAU)>0
          % side check, *all* fields must be identical for 'same' rows (eg
          % including roi, nnbormask etc)
          fldsshared = intersect(tblflds(tblAU),tblflds(tblMD));
          assert(isequaln(tblAU(tfsameAU,fldsshared),...
                          tblMD(locsame(tfsameAU),fldsshared)));
        end
      else
        tfsameAU = false(height(tblAU),1);        
      end
      
      % Types of rows:
      % new row: new MFT
      % existing row/same: existing MFT, matching pAbs/tfocc (and
      %   everything else via side check)
      % existing row/diff: existing MFT, new lbls

      tfnewAU = ~tfexistAU;
      tfdiffAU = tfexistAU & ~tfsameAU;
      locdiffidxsonly = locexist(tfdiffAU);
      
      % remove the existing/diff rows and add the new and diff rows.
      % we used to add new rows and update existing/diff rows, but that
      % update is nontrivial as we crop/rotate plbls here. theoretically
      % that could be factored out of the add but keep it simple for now.
      obj.dat.rmRows(locdiffidxsonly); 
      tblAUadd = tblAU(tfnewAU | tfdiffAU,:);
      tblAddReadFailed = obj.add(tblAUadd,lObj,'wbObj',wbObj,...
        'prmsTgtCrop',prmsTgtCrop);      
      [tfAU,locAU] = tblismember(tblAU,obj.dat.MD,MFTable.FLDSID);
      
      if verbose
        nNew = nnz(tfnewAU);
        nDiff = nnz(tfdiffAU);
        nSame = nnz(tfsameAU);
        fprintf(1,'ppdb addAndUpdate. %d/%d/%d new/diff/same rows.\n',...
          nNew,nDiff,nSame);
      end
    end
   
    function pAbs = invtform(obj,tMFT,pRel)
      % Inverse transform arbitrary data (eg classify results) based on 
      % MD.tformA
      %
      % tMFT: [npred x ncol] Table with MFTable.FLDSID
      % pRel: [npred x npts x nviews x d x nsets] 'Relative' landmark posn data. nsets
      %   is an arbitrary dimension for vectorization
      
      tMD = obj.dat.MD;
      [tf,loc] = tblismember(tMFT,tMD,MFTable.FLDSID); 

      [n,npts,nvw,d,nsets] = size(pRel);
      assert(d==2);
      pAbs = nan(n,npts,nvw,d,nsets);
      for ivw=1:nvw
        tMDformAvw = tMD.tformA(:,(1:6)+(ivw-1)*6); 
        % reconstitute tform matrix
        tMDformAvw(:,end+1:end+3) = repmat([0 0 1],size(tMDformAvw,1),1);
        for i=1:n
          if ~tf(i)
            warningNoTrace('No transform matrix found for mov=%d,frm=%d,tgt=%d.',...
              tMFT.mov(i),tMFT.frm(i),tMFT.iTgt(i));
          else
            idxMD = loc(i);
            A = reshape(tMDformAvw(idxMD,:),3,3);
            tform = maketform('affine',A);
            [pAbs(i,:,ivw,1,:),pAbs(i,:,ivw,2,:)] = ...
              tforminv(tform,pRel(i,:,ivw,1,:),pRel(i,:,ivw,2,:));
          end
        end
      end      
    end
  end
  
%     function updateLabels(obj,tblUp,lObj,varargin)
%       % Update rows, labels (pGT and tfocc) ONLY. images don't change!
%       %
%       % tblUp: updated rows (rows with updated pGT/tfocc).
%       %   MFTable.FLDSCORE fields are required. Only .pGT and .tfocc are 
%       %   otherwise used. Other fields ignored, INCLUDING eg .roi and 
%       %   .nNborMask. Ie, you cannot currently update the roi of a row in 
%       %   the cache (whose image has already been fetched)
%       
%       [prmpp,updateRowsMustMatch] = myparse(varargin,...
%         'prmpp',[],... % preprocessing params
%         'updateRowsMustMatch',false ... % if true, assert/check that tblUp matches current data
%         );
% 
%       if isempty(prmpp)
%         prmpp = lObj.preProcParams;
%         if isempty(prmpp)
%           error('Please specify preprocessing parameters.');
%         end
%       end
% 
%       dataCurr = obj.dat;
%       
%       nUpdate = size(tblUp,1);
%       if nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB 
%                    % table indexing API may not be polished
%         [tf,loc] = tblismember(tblUp,dataCurr.MD,MFTable.FLDSID);
%         assert(all(tf));
%         if updateRowsMustMatch
%           assert(isequal(dataCurr.MD{loc,'tfocc'},tblUp.tfocc),...
%             'Unexpected discrepancy in preproc data cache: .tfocc field');
%           if tblfldscontains(tblUp,'roi')
%             assert(isequal(dataCurr.MD{loc,'roi'},tblUp.roi),...
%               'Unexpected discrepancy in preproc data cache: .roi field');
%           end
%           if tblfldscontains(tblUp,'nNborMask')
%             assert(isequal(dataCurr.MD{loc,'nNborMask'},tblUp.nNborMask),...
%               'Unexpected discrepancy in preproc data cache: .nNborMask field');
%           end
%           assert(isequaln(dataCurr.pGT(loc,:),tblUp.p),...
%             'Unexpected discrepancy in preproc data cache: .p field');
%         else
%           fprintf(1,'Updating labels for %d rows...\n',nUpdate);
%           dataCurr.MD{loc,'tfocc'} = tblUp.tfocc; % AL 20160413 throws if nUpdate==0
%           dataCurr.pGT(loc,:) = tblUp.p;
%           % Check .roi, .nNborMask?
%         end
%         
%         obj.tsLastEdit = now;
%       end
%     end

end