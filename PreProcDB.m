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
        
    function [tblNewReadFailed,dataNew] = add(obj,tblNew,lObj,varargin)
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
      % tblNewReadFailed: table of failed-to-read rows. Currently subset of
      %   tblNew. If non-empty, then .dat was not updated with these rows 
      %   as requested.
      % dataNew: CPRData of new data created/added
      
      [wbObj,prmpp,computeOnly] = myparse(varargin,...
        'wbObj',[],...
        'prmpp',[],... % preprocessing params
        'computeOnly',false... % if true, don't update the DB, compute only.
        );

      tfWB = ~isempty(wbObj);
      if isempty(prmpp)
        prmpp = lObj.preProcParams;
        if isempty(prmpp)
          error('Please specify preprocessing parameters.');
        end
      end
      
      assert(isstruct(prmpp),'Expected parameters to be struct/value class.');

      FLDSREQUIRED = [MFTable.FLDSCORE 'pAbs'];
      FLDSALLOWED = [MFTable.FLDSCORE {'roi' 'nNborMask'}];
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
      
      if prmpp.histeq
        warningNoTrace('Histogram Equalization currently disabled for Deep Learning trackers.');
        prmpp.histeq = false;
      end
      if prmpp.BackSub.Use
        warningNoTrace('Background subtraction currently disabled for Deep Learning trackers.');
        prmpp.BackSub.Use = false;
      end
      if prmpp.NeighborMask.Use
        warningNoTrace('Neighbor masking currently disabled for Deep Learning trackers.');
        prmpp.NeighborMask.Use = false;
      end
      assert(isempty(prmpp.channelsFcn));
                        
      tblNewConc = lObj.mftTableConcretizeMov(tblNew);
      nNew = height(tblNew);
      if nNew>0
        fprintf(1,'Adding %d new rows to data...\n',nNew);
        
        [I,nNborMask,didread,tformA] = CPRData.getFrames(tblNewConc,...
          'wbObj',wbObj,...
          'forceGrayscale',lObj.movieForceGrayscale,...
          'preload',lObj.movieReadPreLoadMovies,...
          'movieInvert',lObj.movieInvert,...          
          'roiPadVal',prmpp.TargetCrop.PadBkgd,...
          'rotateImsUp',prmpp.TargetCrop.AlignUsingTrxTheta,...
          'isDLpipeline',true,...
          'doBGsub',prmpp.BackSub.Use,...
          'bgReadFcn',prmpp.BackSub.BGReadFcn,...
          'bgType',prmpp.BackSub.BGType,...
          'maskNeighbors',prmpp.NeighborMask.Use,...
          'maskNeighborsMeth',prmpp.NeighborMask.SegmentMethod,...
          'maskNeighborsEmpPDF',lObj.fgEmpiricalPDF,...
          'fgThresh',prmpp.NeighborMask.FGThresh,...
          'trxCache',lObj.trxCache,...
          'labeler',lObj,...
          'usePreProcData',lObj.copyPreProcData);
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
        
        % AL: a little worried if all reads fail -- might get a harderr
        
        tfColsAllowed = ismember(tblNew.Properties.VariableNames,FLDSALLOWED);
        tblNewMD = tblNew(:,tfColsAllowed);
        tblNewMD = [tblNewMD table(nNborMask)];
        
        pAbs = tblNewMD.p;
        n = height(tblNewMD);
        nPhysPts = lObj.nPhysPoints;
        nView = lObj.nview;
        pRel = reshape(pAbs,[n nPhysPts nView 2]);
        szassert(tformA,[3 3 n nView]);
        for i=1:n
          for ivw=1:nView
            tform = maketform('affine',tformA(:,:,i,ivw));
            [pRel(i,:,ivw,1),pRel(i,:,ivw,2)] = ...
              tformfwd(tform,pRel(i,:,ivw,1),pRel(i,:,ivw,2));
          end
        end
        pRel = reshape(pRel,[n nPhysPts*nView*2]);
        tblNewMD.p = pRel;
        
        dataNew = CPRData(I,tblNewMD);
        
        if ~computeOnly
          obj.dat.append(dataNew);
          obj.tsLastEdit = now;
        end
      end
    end
    
    function updateLabels(obj,tblUp,lObj,varargin)
      % Update rows, labels (pGT and tfocc) ONLY. images don't change!
      %
      % tblUp: updated rows (rows with updated pGT/tfocc).
      %   MFTable.FLDSCORE fields are required. Only .pGT and .tfocc are 
      %   otherwise used. Other fields ignored, INCLUDING eg .roi and 
      %   .nNborMask. Ie, you cannot currently update the roi of a row in 
      %   the cache (whose image has already been fetched)
      
      [prmpp,updateRowsMustMatch] = myparse(varargin,...
        'prmpp',[],... % preprocessing params
        'updateRowsMustMatch',false ... % if true, assert/check that tblUp matches current data
        );

      if isempty(prmpp)
        prmpp = lObj.preProcParams;
        if isempty(prmpp)
          error('Please specify preprocessing parameters.');
        end
      end

      dataCurr = obj.dat;
      
      nUpdate = size(tblUp,1);
      if nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB 
                   % table indexing API may not be polished
        [tf,loc] = tblismember(tblUp,dataCurr.MD,MFTable.FLDSID);
        assert(all(tf));
        if updateRowsMustMatch
          assert(isequal(dataCurr.MD{loc,'tfocc'},tblUp.tfocc),...
            'Unexpected discrepancy in preproc data cache: .tfocc field');
          if tblfldscontains(tblUp,'roi')
            assert(isequal(dataCurr.MD{loc,'roi'},tblUp.roi),...
              'Unexpected discrepancy in preproc data cache: .roi field');
          end
          if tblfldscontains(tblUp,'nNborMask')
            assert(isequal(dataCurr.MD{loc,'nNborMask'},tblUp.nNborMask),...
              'Unexpected discrepancy in preproc data cache: .nNborMask field');
          end
          assert(isequaln(dataCurr.pGT(loc,:),tblUp.p),...
            'Unexpected discrepancy in preproc data cache: .p field');
        else
          fprintf(1,'Updating labels for %d rows...\n',nUpdate);
          dataCurr.MD{loc,'tfocc'} = tblUp.tfocc; % AL 20160413 throws if nUpdate==0
          dataCurr.pGT(loc,:) = tblUp.p;
          % Check .roi, .nNborMask?
        end
        
        obj.tsLastEdit = now;
      end
    end

    function [tblAddReadFailed,tfAU,locAU] = addAndUpdate(obj,tblAU,lObj,varargin)
      % Combo of add/updateLabels
      %
      % tblAU: ("tblAddUpdate")
      %   - MFTable.FLDSCORE: required.
      %   - .roi: optional, USED WHEN PRESENT. (prob needs to be either
      %   consistently there or not-there for a given obj or initData()
      %   "session"
      %   IMPORTANT: if .roi is present, .p (labels) are expected to be 
      %   relative to the roi.
      %   - .pTS: optional (if present, deleted)
      %
      % tblAddReadFailed: tbl of failed-read adds. will have height 0 if
      %   everything went well.
      % tfAU: [tfAU,locAU] = tblismember(tblAU,obj.dat.MD,MFTable.FLDSID)
      % locAU: 
      
      [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
        'wbObj',[],... % WaitBarWithCancel. If cancel, obj unchanged.
        'updateRowsMustMatch',false,... % See updateLabels
        'prmpp',[] ...
        );
      
      [tblPnew,tblPupdate] = obj.dat.tblPDiff(tblAU);
      tblAddReadFailed = obj.add(tblPnew,lObj,'wbObj',wbObj,'prmpp',prmpp);
      obj.updateLabels(tblPupdate,lObj,...
        'updateRowsMustMatch',updateRowsMustMatch);
      
      [tfAU,locAU] = tblismember(tblAU,obj.dat.MD,MFTable.FLDSID);
    end
    
  end

end