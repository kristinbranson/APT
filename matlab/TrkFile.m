classdef TrkFile < dynamicprops
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
    listfile_fns = {'pred_locs','to_track','pred_tag','list_file','pred_ts','pred_conf'};
  end
  
  properties
    pTrk = TrkFile.unsetVal;     % [npttrked x 2 x nfrm x ntgt], like labeledpos
    pTrkTS = TrkFile.unsetVal;   % [npttrked x nfrm x ntgt], liked labeledposTS
    pTrkTag = TrkFile.unsetVal;  % [npttrked x nfrm x ntgt] logical, like labeledposTag
    pTrkiPt = TrkFile.unsetVal;  % [npttrked]. point indices labeling rows of .pTrk*. If 
                                 %  npttrked=labeled.nLabelPoints, then .pTrkiPt=1:npttrked.
    pTrkFrm = TrkFile.unsetVal;  % [nfrm]. frames tracked
    pTrkiTgt = TrkFile.unsetVal; % [ntgt]. targets (1-based Indices) tracked

    pTrkFull = TrkFile.unsetVal; % [npttrked x 2 x nRep x nTrkFull], full tracking with replicates
    pTrkFullFT = TrkFile.unsetVal; % [nTrkFull x ncol] Frame-Target table labeling 4th dim of pTrkFull
    
    trkInfo % "user data" for tracker
    % which video these trajectories correspond to
    movfile = TrkFile.unsetVal;
  end
  
  methods
    
    function obj = TrkFile(ptrk,varargin)
      % ptrk: must be supplied, to set .pTrk
      % varargin: p-v pairs for remaining props
      %
      % Example: tfObj = TrkFile(myTrkPos,'pTrkTS',myTrkTS);
      
      if nargin==0        
        return;
      end
      
      nArg = numel(varargin);
      for i=1:2:nArg
        prop = varargin{i};
        val = varargin{i+1};
        if ~isprop(obj,prop)
          warningNoTrace('Adding TrkFile property ''%s''.',prop);
          obj.addprop(prop);
        end         
        obj.(prop) = val;
      end
      
      % trk files can either be full matrices or sparse tables      
      if isnumeric(ptrk),
        obj.pTrk = ptrk;
      elseif isstruct(ptrk),
        obj.InitializeTable(ptrk);
      end
      
      [npttrk,d,nfrm,ntgt] = size(obj.pTrk);
      if d~=2
        error('TrkFile:TrkFile','Expected d==2.');
      end

      if isequal(obj.pTrkTS,TrkFile.unsetVal)
        obj.pTrkTS = zeros(npttrk,nfrm,ntgt);
      end
      validateattributes(obj.pTrkTS,{'numeric'},{'size' [npttrk nfrm ntgt]},'','pTrkTS');
      
      if isequal(obj.pTrkTag,TrkFile.unsetVal)
        obj.pTrkTag = false(npttrk,nfrm,ntgt);
      end
      validateattributes(obj.pTrkTag,{'logical'},...
        {'size' [npttrk nfrm ntgt]},'','pTrkTag');
      
      if isequal(obj.pTrkiPt,TrkFile.unsetVal)
        obj.pTrkiPt = 1:npttrk;
      end
      validateattributes(obj.pTrkiPt,{'numeric'},...
        {'vector' 'numel' npttrk 'positive' 'integer'},'','pTrkiPt');
      
      if isequal(obj.pTrkFrm,TrkFile.unsetVal)
        obj.pTrkFrm = 1:nfrm;
      end
      assert(issorted(obj.pTrkFrm));
      validateattributes(obj.pTrkFrm,{'numeric'},...
        {'vector' 'numel' nfrm 'positive' 'integer'},'','pTrkFrm');

      if isequal(obj.pTrkiTgt,TrkFile.unsetVal)
        obj.pTrkiTgt = 1:ntgt;
      end
      assert(issorted(obj.pTrkiTgt));
      validateattributes(obj.pTrkiTgt,{'numeric'},...
        {'vector' 'numel' ntgt 'positive' 'integer'},'','pTrkiTgt');
      
      tfUnsetTrkFull = isequal(obj.pTrkFull,TrkFile.unsetVal);
      tfUnsetTrkFullFT = isequal(obj.pTrkFullFT,TrkFile.unsetVal);
      assert(tfUnsetTrkFull==tfUnsetTrkFullFT);
      if tfUnsetTrkFull
        obj.pTrkFull = zeros(npttrk,2,0,0);
        obj.pTrkFullFT = table(nan(0,1),nan(0,1),'VariableNames',{'frm' 'iTgt'});
      end
      nRep = size(obj.pTrkFull,3);
      nFull = size(obj.pTrkFull,4);
      validateattributes(obj.pTrkFull,{'numeric'},...
        {'size' [npttrk 2 nRep nFull]},'','pTrkFull');
      assert(istable(obj.pTrkFullFT) && height(obj.pTrkFullFT)==nFull);
    end
    
    % initialize fields from sparse table trk file
    function InitializeTable(obj,s)

      [movfiles,nviewstrack] = TrkFile.convertJSONCellMatrix(s.to_track.movieFiles);

      n = size(s.pred_locs,1);
      npts = size(s.pred_locs,2);
      d = size(s.pred_locs,3);
      assert(d == 2);

      iTgt = nan(n,1);
      frm = nan(n,1);
      movidx = false(n,1);
      off = 0;
      movi = find(strcmp(obj.movfile,movfiles));
      assert(numel(movi)==1);
      for i = 1:numel(s.to_track.toTrack),
        
        x = s.to_track.toTrack{i};
        if double(x{1}) ~= movi,
          continue;
        end
        
        frm0 = x{3};
        if iscell(frm0),
          frm0 = frm0{1};
        end
        frm0 = double(frm0);
        if numel(x{3}) > 1,
          frm1 = x{3}(2);
          if iscell(frm1),
            frm1 = frm1{1};
          end
          frm1 = double(frm1);
          frm1 = frm1-1; % python range
        else
          frm1 = frm0;
        end
        nfrm = frm1-frm0+1;
        frm(off+1:off+nfrm) = frm0:frm1;
        iTgt(off+1:off+nfrm) = double(x{2});
        movidx(off+1:off+nfrm) = true;
        off = off + nfrm;
      end
      % pTrk is [npttrked x 2 x nfrm x ntgt]
      % p = reshape(p,[npts*d nF nTgt]);
      % pcol = p(:,i,j);
      % pred_locs is n x npts x 2
      
      nidx = nnz(movidx);
      nfrm = max(frm);
      [obj.pTrkiTgt,~,tgtidx] = unique(iTgt);
      ntgt = numel(obj.pTrkiTgt);
      pTrk = s.pred_locs(movidx,:,:); %#ok<PROPLC>
      obj.pTrk = nan([npts,2,nfrm,ntgt]);
      ists = isfield(s,'pred_ts');
      istag = isfield(s,'pred_tag');
      if ists,
        obj.pTrkTS = zeros(npttrk,nfrm,ntgt);
      end
      if istag,
        obj.pTrkTag = false(npttrk,nfrm,ntgt);
      end

      for i = 1:nidx,
        obj.pTrk(:,:,frm(i),tgtidx(i)) = permute(pTrk(i,:,:),[2,3,1]); %#ok<PROPLC>
        if ists,
          obj.pTrkTS(:,frm(i),tgtidx(i)) = s.pred_ts(:,i);
        end
        if istag,
          obj.pTrkTag(:,frm(i),tgtidx(i)) = s.pred_tag(:,i);
        end
      end
      
%       pTrk = reshape(pTrk,[nidx,npts*2]); %#ok<PROPLC>
%       tfOcc = false(nidx,npts);
%       pTrkTS = nan(n,npts); %#ok<PROPLC>
%       obj.pTbl = table(frm,iTgt,pTrk,tfOcc,pTrkTS); %#ok<PROPLC>
    end
    
%     function v = isTable(obj)
%       v = strcmpi(obj.type,'table');
%     end
% 
%     function v = isFullMatrix(obj)
%       v = strcmpi(obj.type,'fullmatrix');
%     end

    
    function save(obj,filename)
      % Saves to filename; ALWAYS OVERWRITES!!
      
      warnst = warning('off','MATLAB:structOnObject');
      s = struct(obj);
      warning(warnst);
      s = rmfield(s,'unsetVal'); %#ok<NASGU>      
      save(filename,'-mat','-struct','s');
    end
    
    function tbl = tableform(obj)
      p = obj.pTrk;
      [npts,d,nF,nTgt] = size(p);
      assert(d==2);
      p = reshape(p,[npts*d nF nTgt]);
      ptag = obj.pTrkTag;  
      pTS = obj.pTrkTS;
      pfrm = obj.pTrkFrm;
      ptgt = obj.pTrkiTgt;
      
      s = struct('frm',cell(0,1),'iTgt',[],'pTrk',[],'tfOcc',[],'pTrkTS',[]);
      for i=1:nF
      for j=1:nTgt
        pcol = p(:,i,j);
        tfOcccol = ptag(:,i,j);
        pTScol = pTS(:,i,j);
        if any(~isnan(pcol)) || any(tfOcccol)
          s(end+1,1).frm = pfrm(i); %#ok<AGROW>
          s(end).iTgt = ptgt(j);
          s(end).pTrk = pcol(:)';
          s(end).tfOcc = tfOcccol(:)';
          s(end).pTrkTS = pTScol(:)';
        end
      end
      end
      
      tbl = struct2table(s);
    end
    
    function mergePartial(obj1,obj2)
      % Merge trkfile into current trkfile. Doesn't merge .pTrkFull* fields 
      % (for now).
      %
      % obj2 TAKES PRECEDENCE when tracked frames/data overlap
      %
      % obj/obj2: trkfile objs. obj.pTrk and obj2.pTrk must have the same
      % size.
      
      assert(isscalar(obj1) && isscalar(obj2));
      %assert(isequal(size(obj.pTrk),size(obj2.pTrk)),'Size mismatch.');
      assert(isequal(obj1.pTrkiPt,obj2.pTrkiPt),'.pTrkiPt mismatch.');
      %assert(isequal(obj.pTrkiTgt,obj2.pTrkiTgt),'.pTrkiTgt mismatch.');
      
      if ~isempty(obj1.pTrkFull) || ~isempty(obj2.pTrkFull)
        warningNoTrace('.pTrkFull contents discarded.');
      end
      
      frmUnion = union(obj1.pTrkFrm,obj2.pTrkFrm);
      iTgtUnion = union(obj1.pTrkiTgt,obj2.pTrkiTgt);
      
      npttrk = numel(obj1.pTrkiPt);
      nfrm = numel(frmUnion);
      ntgt = numel(iTgtUnion);
      
      % lots of changes because we might have non-dense results
      % and we want to use the newest tracking on a per frame & target
      % basis
      % old code used to assume that obj2 was all newer than obj1
      nfrmConflict = zeros(1,ntgt);

      % sizes of fields that might be in the trkfile objects      
      szs = struct;
      szs.pTrk = [npttrk,2,nfrm,ntgt];
      szs.pTrkTS = [npttrk,nfrm,ntgt];
      szs.pTrkTag = [npttrk,nfrm,ntgt];
      szs.pTrk3d = [npttrk 3 nfrm ntgt];
      szs.pTrkSingleView = [npttrk 2 nfrm ntgt];
      szs.pTrkconf = [npttrk nfrm ntgt];
      szs.pTrkconf_unet = [npttrk nfrm ntgt];
      szs.pTrklocs_mdn = [npttrk 2 nfrm ntgt];
      szs.pTrklocs_unet = [npttrk 2 nfrm ntgt];
      szs.pTrkocc = [npttrk nfrm ntgt];
      flds = fieldnames(szs);

      % figure out which frames to copy from each object
      allidx = cell(ntgt,2);
      allnewidx = cell(ntgt,2);
      allitgt = nan(ntgt,2);
      
      for itgt = 1:ntgt,
      
        itgt1 = find(itgt==obj1.pTrkiTgt,1);
        itgt2 = find(itgt==obj2.pTrkiTgt,1);
        % frames that have data for each object
        if isempty(itgt1),
          idx1 = false(size(obj1.pTrk,3),1);
        else
          idx1 = squeeze(~isnan(obj1.pTrk(1,1,:,itgt1)));
        end
        if isempty(itgt2),
          idx2 = false(size(obj2.pTrk,3),1);
        else
          idx2 = squeeze(~isnan(obj2.pTrk(1,1,:,itgt2)));
        end
        
        % new indices for these data
        frm1 = obj1.pTrkFrm(idx1);
        frm2 = obj2.pTrkFrm(idx2);
        [~,newidx1] = ismember(frm1,frmUnion);
        [~,newidx2] = ismember(frm2,frmUnion);

        % which data is newer -- assuming all parts tracked together and
        % have same timestamp
        newts = zeros(1,nfrm);
        assert(isempty(itgt2) || all(all(obj2.pTrkTS(1,idx2,itgt2)==obj2.pTrkTS(2:end,idx2,itgt2))));
        assert(isempty(itgt1) || all(all(obj1.pTrkTS(1,idx1,itgt1)==obj1.pTrkTS(2:end,idx1,itgt1))));
        if ~isempty(itgt1),
          newts(newidx1) = obj1.pTrkTS(1,idx1,itgt1);
        end
        isassigned = newts(newidx2)~=0;
        isnewer = ~isassigned;
        if ~isempty(itgt2),
          isnewer = isnewer | ...
            newts(newidx2) < obj2.pTrkTS(1,idx2,itgt2);
        end
        nfrmConflict(itgt) = nnz(isassigned);
        idx2 = find(idx2);
        idx2newer = idx2(isnewer);
        newidx2newer = newidx2(isnewer);

        allidx{itgt,1} = idx1;
        allidx{itgt,2} = idx2newer;
        allnewidx{itgt,1} = newidx1;
        allnewidx{itgt,2} = newidx2newer;
        if ~isempty(itgt1),
          allitgt(itgt,1) = itgt1;
        end
        if ~isempty(itgt2),
          allitgt(itgt,2) = itgt2;
        end
      end
      
      for fldi = 1:numel(flds),
        obj1.hlpMergePartialTgts(obj2,flds{fldi},szs.(flds{fldi}),allidx,allnewidx,allitgt);
      end
        
% old code
%       tfobj1HasRes = false(nfrm,1);
%       tfobj2HasRes = false(nfrm,1);
%       [~,locfrm1] = ismember(obj1.pTrkFrm,frmUnion);
%       [~,locfrm2] = ismember(obj2.pTrkFrm,frmUnion);
%       [~,loctgt1] = ismember(obj1.pTrkiTgt,iTgtUnion);
%       [~,loctgt2] = ismember(obj2.pTrkiTgt,iTgtUnion);          
%       tfobj1HasRes(locfrm1,loctgt1) = true;
%       tfobj2HasRes(locfrm2,loctgt2) = true;
%       tfConflict = tfobj1HasRes & tfobj2HasRes;
%       nfrmConflict = nnz(any(tfConflict,2));
%       ntgtConflict = nnz(any(tfConflict,1));
%       if nfrmConflict>0 % =>ntgtConflict>0
%         warningNoTrace('TrkFiles share common results for %d frames, %d targets. Second trkfile will take precedence.',...
%           nfrmConflict,ntgtConflict);
%       end
%      
%       % init new pTrk, pTrkTS, pTrkTag; write results1, then results2 
%       pTrk = nan(npttrk,2,nfrm,ntgt);
%       pTrkTS = nan(npttrk,nfrm,ntgt);
%       pTrkTag = nan(npttrk,nfrm,ntgt);            
%       pTrk(:,:,locfrm1,loctgt1) = obj1.pTrk;      
%       pTrk(:,:,locfrm2,loctgt2) = obj2.pTrk;
%       pTrkTS(:,locfrm1,loctgt1) = obj1.pTrkTS;
%       pTrkTS(:,locfrm2,loctgt2) = obj2.pTrkTS;
%       pTrkTag(:,locfrm1,loctgt1) = obj1.pTrkTag;
%       pTrkTag(:,locfrm2,loctgt2) = obj2.pTrkTag;
% 
%       obj1.pTrk = pTrk;
%       obj1.pTrkTS = newts;
%       obj1.pTrkTag = pTrkTag;      
%       % Could use hlpMergePartial in the above 
%       
%       obj1.hlpMergePartial(obj2,'pTrk3d',[npttrk 3 nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrkSingleView',[npttrk 2 nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrkconf',[npttrk nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrkconf_unet',[npttrk nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrklocs_mdn',[npttrk 2 nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrklocs_unet',[npttrk 2 nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
%       obj1.hlpMergePartial(obj2,'pTrkocc',[npttrk nfrm ntgt],locfrm1,loctgt1,locfrm2,loctgt2);
% 
%       %obj1.pTrkiPt = obj1.pTrkiPt; unchanged
      obj1.pTrkFrm = frmUnion;
      obj1.pTrkiTgt = iTgtUnion;
      obj1.pTrkFull = [];
      obj1.pTrkFullFT = [];
      if sum(nfrmConflict)>0 % =>ntgtConflict>0
        warningNoTrace('TrkFiles share common results for %d frames across %d targets. Newest results will take precedence.',...
          sum(nfrmConflict),nnz(nfrmConflict));
      end
      
      if iscell(obj1.trkInfo)
        obj1.trkInfo{end+1} = obj2.trkInfo;
      else
        obj1.trkInfo = {obj1.trkInfo obj2.trkInfo};
      end
    end
    
    function hlpMergePartialTgts(obj1,obj2,fld,valsz,allidx,allnewidx,allitgt)
      % helper for mergePartial for each property
      % obj1.hlpMergePartialTgts(obj2,fld,valsz,allidx,allnewidx,allitgt)
      %
      % mutates obj1.(fld)
      % newidx2 overwrites newidx1
      isprop1 = isprop(obj1,fld);
      isprop2 = isprop(obj2,fld);
      
      if ~(isprop1 || isprop2),
        return;
      end
      
      ntgts = size(allitgt,1);
      
      val = nan(valsz);
      valndim = numel(valsz);
      for itgt = 1:ntgts,
        itgt1 = allitgt(itgt,1);
        itgt2 = allitgt(itgt,2);
        if isprop1 && ~isnan(itgt1),
          idx1 = allidx{itgt,1};
          newidx1 = allnewidx{itgt,1};
          switch valndim
            case 3
              val(:,newidx1,itgt) = obj1.(fld)(:,idx1,itgt1);
            case 4
              val(:,:,newidx1,itgt) = obj1.(fld)(:,:,idx1,itgt1);
            otherwise
              assert(false);
          end
        end
        
        if isprop2 && ~isnan(itgt2),
          idx2 = allidx{itgt,2};
          newidx2 = allnewidx{itgt,2};
          switch valndim
            case 3
              val(:,newidx2,itgt) = obj2.(fld)(:,idx2,itgt2);
            case 4
              val(:,:,newidx2,itgt) = obj2.(fld)(:,:,idx2,itgt2);
            otherwise
              assert(false);
          end
        end
      end
        
      if ~isprop1
        obj1.addprop(fld);
      end
      obj1.(fld) = val;
    end
    
    function hlpMergePartial(obj1,obj2,fld,valsz,locfrm1,loctgt1,locfrm2,loctgt2)
      % helper for mergePartial to handle additional/dynamic props
      %
      % mutates obj1.(fld)
      % obsolete? 
      
      isprop1 = isprop(obj1,fld);
      isprop2 = isprop(obj2,fld);
      
      if isprop1 || isprop2
        val = nan(valsz);
        valndim = numel(valsz);
        if isprop1
          switch valndim
            case 3
              val(:,locfrm1,loctgt1) = obj1.(fld);
            case 4
              val(:,:,locfrm1,loctgt1) = obj1.(fld);
            otherwise
              assert(false);
          end
        end
        
        if isprop2
          switch valndim
            case 3
              val(:,locfrm2,loctgt2) = obj2.(fld);
            case 4              
              val(:,:,locfrm2,loctgt2) = obj2.(fld);
            otherwise
              assert(false);
          end
        end
        
        if ~isprop1
          obj1.addprop(fld);
        end        
        obj1.(fld) = val;
      end
    end
    
    function indexInPlace(obj,ipts,ifrms,itgts)
      % Subscripted-index a TrkFile *IN PLACE*. Your TrkFile will become
      % smaller and you will 'lose' data!
      % 
      % ipts: [nptsidx] actual/absolute landmark indices; will be compared 
      %   against .pTrkiPt. Can be -1 indicating "all available points".
      % ifrms: [nfrmsidx] actual frames; will be compared against .pTrkFrm.
      %   Can be -1 indicating "all available frames"
      % itgts: [ntgtsidx] actual targets; will be compared against
      % .pTrkiTgt. Can be -1 indicating "all available targets"
      %
      % Postconditions: All properties of TrkFile, including
      % dynamic/'extra' properties, are indexed appropriately to the subset
      % as specified by ipts, ifrms, itgts. 
      
      assert(isscalar(obj),'Obj must be a scalar Trkfile object.');
      
      if isequal(ipts,-1)
        ipts = obj.pTrkiPt;
      end
      if isequal(ifrms,-1)
        ifrms = obj.pTrkFrm;
      end
      if isequal(itgts,-1)
        itgts = obj.pTrkiTgt;
      end
      [tfpts,ipts] = ismember(ipts,obj.pTrkiPt);
      [tffrms,ifrms] = ismember(ifrms,obj.pTrkFrm);
      [tftgts,itgts] = ismember(itgts,obj.pTrkiTgt);
      if ~( all(tfpts) && all(tffrms) && all(tftgts) )
        error('All specified points, frames, and targets are not present in TrkFile.');
      end
      
      szpTrk = size(obj.pTrk); % [npts x 2 x nfrm x ntgt]
      szpTrkTS = size(obj.pTrkTS); % [npts x nfrm x ntgt]
      propsDim1Only = {'pTrkFull'};
      
      props = properties(obj);
      for p=props(:)',p=p{1};
        
        v = obj.(p);
        szv = size(v);
        
        if strcmp(p,'pTrkiPt')
          obj.(p) = v(ipts);
        elseif strcmp(p,'pTrkFrm')
          obj.(p) = v(ifrms);
        elseif strcmp(p,'pTrkiTgt')
          obj.(p) = v(itgts);
        elseif any(strcmp(p,propsDim1Only))
          szvnew = szv;
          szvnew(1) = numel(ipts);
          v = reshape(v,szv(1),[]);
          v = v(ipts,:);
          v = reshape(v,szvnew);
          obj.(p) = v;
        elseif isnumeric(v) || islogical(v)
          if isequal(szv,szpTrk)
            obj.(p) = v(ipts,:,ifrms,itgts);
          elseif isequal(szv,szpTrkTS)
            obj.(p) = v(ipts,ifrms,itgts);
          else
            warningNoTrace('Numeric property ''%s'' with unrecognized shape: %s',...
              p,mat2str(szv));
          end
        else
          % non-numeric prop, no action
        end
      end
    end
  end
  
  methods (Static)

    % v = isValidLoadFullMatrix(s)
    % whether the struct resulting from loading is a full matrix trk file
    function v = isValidLoadFullMatrix(s)
      v = isfield(s,'pTrk');
    end
    
    % v = isValidLoadTable(s)
    % whether the struct resulting from loading is a table trk file
    function v = isValidLoadTable(s)
      v = all(isfield(s,{'to_track','pred_locs'}));
    end      
    
    function v = isValidLoad(s)
      v = isValidLoadFullMatrix(s) || isValidLoadTable(s);
    end
    
    function filetype = getFileType(s)
      
      if all(isfield(s,{'startframes' 'endframes'}))
        filetype = 'tracklet';
      elseif TrkFile.isValidLoadFullMatrix(s),
        filetype = 'fullmatrix';
      elseif TrkFile.isValidLoadTable(s),
        filetype = 'table';
      else
        filetype = '';
      end
    end
    
    function trkfileObj = load(filename,movfile,issilent)

      ntries = 5;
      if nargin < 2,
        movfile = '';
      end
      if nargin < 3,
        issilent = false;
      end
      if ~exist(filename,'file'),
        trkfileObj = [];
        return;
      end
      
      for tryi = 1:ntries,
        s = load(filename,'-mat');
        if isempty(fieldnames(s)) && tryi < ntries,
          fprintf('Attempt %d to load %s failed, retrying\n',tryi,filename);
          pause(5);
          continue;
        end
        break;
      end
      filetype = TrkFile.getFileType(s);
      if strcmp(filetype,'tracklet')
        %%% Tracklet early return %%%
        trkfileObj = load_tracklet(s);
        % We do this right here, upon entry into APT, but this might more
        % properly be done further inside the App (eg at vizInit-time) as 
        % .x, .y are more for viz purposes.
        trkfileObj = TrxUtil.ptrxAddXY(trkfileObj); 
        [trkfileObj.movfile] = deal(movfile);
        return;        
      end        
        
      s = TrkFile.modernizeStruct(s);
      if isempty(filetype),
        error('TrkFile:load',...
          'File ''%s'' is not a valid saved trkfile structure.',filename);
      end
      
      if issilent,
        mc = meta.class.fromName('TrkFile');
        propnames = {mc.PropertyList.Name}';
        fns = fieldnames(s);%setdiff(fieldnames(s),TrkFile.listfile_fns);
        tfrecog = ismember(fns,propnames);
        fnsunrecog = fns(~tfrecog);
        srecog = rmfield(s,fnsunrecog);
      else
        srecog = s;
      end
      
      switch filetype,
        case 'fullmatrix',
          pTrk = s.pTrk;
          pvs = struct2pvs(rmfield(srecog,'pTrk'));
        case 'table'
          pTrk = struct;
          pTrk.pred_locs = s.pred_locs;
          pTrk.to_track = s.to_track;
          pvs = struct2pvs(srecog);
      end
      trkfileObj = TrkFile(pTrk,'movfile',movfile,pvs{:});
      
      if issilent,
        fnsunrecog = setdiff(fnsunrecog,TrkFile.listfile_fns);
        for f=fnsunrecog(:)',f=f{1};
          trkfileObj.addprop(f);
          trkfileObj.(f) = s.(f);
        end
      end
    end
    
    function trkfileObj = loadsilent(filename,movfile)

      if nargin < 2,
        movfile = '';
      end
      trkfileObj = TrkFile.load(filename,movfile,true);
      
%       % ignore fields of struct that aren't TrkFile props. For 3rd-party
%       % generated Trkfiles
%       s = load(filename,'-mat');
%       s = TrkFile.modernizeStruct(s);      
%       
%       pTrk = s.pTrk;
%       
%       mc = meta.class.fromName('TrkFile');
%       propnames = {mc.PropertyList.Name}';
%       fns = fieldnames(s);
%       tfrecog = ismember(fns,propnames);
%       %fnsrecog = fns(tfrecog);
%       fnsunrecog = fns(~tfrecog);
%       
%       srecog = rmfield(s,fnsunrecog);
%       pvs = struct2pvs(rmfield(srecog,'pTrk'));
%       trkfileObj = TrkFile(pTrk,pvs{:});
%       
%       for f=fnsunrecog(:)',f=f{1};
%         trkfileObj.addprop(f);
%         trkfileObj.(f) = s.(f);
%       end      
    end

    function s = modernizeStruct(s)

      if isfield(s,'pred_conf'),
        s = rmfield(s,'pred_conf');
      end
      if isfield(s,'list_file'),
        s = rmfield(s,'list_file');
      end
      if isfield(s,'listfile_fns')
        s = rmfield(s,'listfile_fns');
      end
      
      % s: struct loaded from trkfile saved to matfile
      if isfield(s,'pTrkTag') && iscell(s.pTrkTag)
        s.pTrkTag = strcmp(s.pTrkTag,'occ');
      end
    end
    
    function [nFramesTracked,didload] = getNFramesTrackedPartFile(tfile)
      
      nFramesTracked = 0;
      didload = false;
      s = readtxtfile(tfile);
      PAT = '(?<numfrmstrked>[0-9]+)';
      toks = regexp(s,PAT,'names','once');
      if isempty(toks),
        return;
      end
      nFramesTracked = str2double(toks{1}.numfrmstrked);

    end

    function [nFramesTracked,didload] = getNFramesTrackedMatFile(tfile)
      
      nFramesTracked = 0;
      didload = false;
      ntries = 5;
      
      for tryi = 1:ntries,
        m = matfile(tfile);
        fns = fieldnames(m);

        if any(strcmp('startframes',fns))
          nFramesTracked = m.pTrkFrm(1,end) - m.pTrkFrm(1,1) + 1;
          didload = true;
        elseif ismember('pTrkFrm',fns)
          nFramesTracked = numel(m.pTrkFrm);
          didload = true;
        elseif ismember('pTrk',fns),
          nd = ndims(m.pTrk);
          if nd == 3,
            nFramesTracked = nnz(~isnan(m.pTrk(1,1,:)));
          else
            nFramesTracked = nnz(~isnan(m.pTrk(1,1,:,:)));
          end
          didload = true;
        elseif ismember('pred_locs',fns),
          nFramesTracked = nnz(~isnan(m.pred_locs(:,1)));
          didload = true;
        elseif ismember('locs',fns)
          % gt mat-file
          % AL: not sure want nnz(~isnan(...)) here; what if a tracker
          % predicted occluded or something, could that mess stuff up?
          nFramesTracked = size(m.locs,1);
          didload = true;
        else
          didload = false;
          nFramesTracked = 0;
          fprintf('try %d, variables in %s:\n',tryi,tfile);
          disp(m);
          pause(5);
        end
        if didload,
          break;
        end
      end

    end

    
    function [nFramesTracked,didload] = getNFramesTracked(tfile)
      nFramesTracked = 0;
      didload = false;
      if ~exist(tfile,'file'),
        return;
      end
      try
        [nFramesTracked,didload] = TrkFile.getNFramesTrackedMatFile(tfile);
      catch
        try
          [nFramesTracked,didload] = TrkFile.getNFramesTrackedPartFile(tfile);
        catch ME,
          warning('Could not read n. frames tracked from %s:\n%s',tfile,getReport(ME));
          didload = false;
        end
      end
%       if didload,
%         fprintf('Read %d frames tracked from %s\n',nFramesTracked,tfile);
%       end
    end
    
    function [x,nc] = convertJSONCellMatrix(xin)
      
      x1 = cell(size(xin));
      for i = 1:numel(xin),
        if iscell(xin{i}),
          x1{i} = cellfun(@char,xin{i},'Uni',0);
        else
          x1{i} = {char(xin{i})};
        end
      end
      nc = cellfun(@numel,x1);
      assert(nc==nc(1));
      nc = nc(1);
      x = cell(numel(x1),nc);
      for i = 1:numel(x1),
        x(i,:) = x1{i};
      end
      
    end
    
  end
  
end
