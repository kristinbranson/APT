classdef TrkFile < dynamicprops
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
    listfile_fns = {'pred_locs','to_track','pred_tag','list_file',...
      'pred_ts','pred_conf'};
   
    % AL 20220620: pTrk* props have shapes like [npt x d x nfrm] (eg: pTrk)
    % or like [npt x nfrm] (eg: pTrkConf). Record properties of the first 
    % type here with their dimension so that empty/new TrkFiles can be 
    % initialized with the proper shaped-empty arrays. 
    %
    % Using [] instead of shaped emptys might be possible in some cases but
    % it is safer to use shaped emptys.
    ptrk_fns_dimensional = struct('pTrk',2,'pTrk3d',3,'pTrkSingleView',2);
  end
  
  properties
    pTrk = TrkFile.unsetVal;     % [npttrked x 2 x nfrm x ntgt], like labeledpos
    pTrkTS = TrkFile.unsetVal;   % [npttrked x nfrm x ntgt], liked labeledposTS
    pTrkTag = TrkFile.unsetVal;  % [npttrked x nfrm x ntgt] logical, like labeledposTag
    pTrkiPt = TrkFile.unsetVal;  % [npttrked]. point indices labeling rows of .pTrk*. If 
                                 %  npttrked=labeled.nLabelPoints, then .pTrkiPt=1:npttrked.
    pTrkFrm = TrkFile.unsetVal;  % [nfrm]. frames tracked
    pTrkiTgt = TrkFile.unsetVal; % [ntgt]. targets (1-based Indices) tracked
                                 % this is NOT used for indexing, e.g.
                                 % getPTrkTgt(i) just gives pTrk(:,:,:,i)

%     pTrkSingleView = TrkFile.unsetVal; % For multi-view stuff                           
    % CPR-specific
    pTrkFull = TrkFile.unsetVal; % [npttrked x 2 x nRep x nTrkFull], full tracking with replicates
    pTrkFullFT = TrkFile.unsetVal; % [nTrkFull x ncol] Frame-Target table labeling 4th dim of pTrkFull
    
    trkInfo % struct. "user data" for tracker
    % which video these trajectories correspond to
    movfile = TrkFile.unsetVal;
    
    isfull = false;  % if true, .pTrk etc are full
    
    % tracklet stuff
    startframes 
    endframes
    nframes = TrkFile.unsetVal;
    
    %frm2tlt  % [nfrmtot x ntgt] logical indicating frames where tracklets 
    %         % are live.
    %frm2tltnnz % nnz(frm2tlt)
    npts
    
    trkfldsextra = {}; % extra fields added
  end
  properties (Dependent)
    ntracklets
    %ntlts  % Wasn't used anywhere
  end
  
  methods 
%     function v = get.npts(obj)
%       % assumes at least one tracklet...
%       v = size(obj.pTrk{1},1);
%     end
%     function v = get.ntlts(obj)
%       v = obj.ntracklets;
%     end
    function v = get.ntracklets(obj)
      if obj.isfull
        v = size(obj.pTrk,4);
      else
        v = numel(obj.pTrk);
      end
    end
  end
  
  % Dev notes 20210514
  %
  % This class prob better just named 'Trk' at this point as it has
  % runtime-related functionality that extends beyond save/load.
  %
  % TrkFiles represent tracking output, while Labels represent user 
  % annotations. Tracking output typically comes in (large) blocks of 
  % contiguous frames for one or more targets; Labels on the other hand
  % are usually quite sparse across movies, time, and targets.
  %
  % *TrkFile storage formats*
  % 1) Dense array. Not bad for single-target or well-body-tracked
  % projects, but inefficient for general multianimal tracking.
  % 2) Sparse array. This is more an implementaiton detail/
  % compression scheme on top of 1. 
  % 3) Tracklet. This is the main production format produced by the APT
  % backend.
  % 4) PTrx. This is similar to 3 but mirrors the Ctrax 'trx' format. It 
  % has backwards compatibility with Ctrax but is hard to work with from Py 
  % and doesn't have a convenient place for top-level metadata.
  % 5) Table/Labels. This tabular format is useful in the case of sparse
  % data and for some analysis tasks. APT can also produce this format in
  % the case of "listfile"/sparse tracking.
  %
  % TrkFile primarily supports 3) as APT's production tracking format, but 
  % also supports formats 1) and 2) as those have similar top-level fields.
  % 
  % 4) and 5) have different storage schemes. A full API for 5) is
  % available in Labels.m; see TrxUtil.m for utilities for 4). 
  %
  % In fact the storage of 5) could be shoehorned to work with this class. 
  % The situation with storage schemes is a little odd. What we/clients 
  % care most about is a standard API for tracking result access. For 
  % this purpose, TrkFile, Labels, and any class of this ilk could 
  % inherit from an ABC or just implement a common interface.
  %
  % The precise storage is to some extent an implementation/optimization
  % detail and whether different formats are stored in a single class like 
  % TrkFile or have separate classes is not necessarily crucial. Since the
  % data fields are common, different storage formats can overlap
  % significantly in their property/fieldnames and hence can be shoehorned
  % into a common class.
  %
  % It is not quite so simple, as the storage format (assuming a public
  % implementation, which would be natural in MATLAB) does imply a 
  % natural/built-in (property-based) API which users will interact with 
  % and write code against. So the choice of re-using TrkFile vs
  % dispatching to different underlying classes is not entirely hidden.
  %
  % Note that there is no single optimal storage format, as eg 'sparse'
  % tracking is not optimally stored as tracklets and vice versa etc. The
  % TrackletTest performance test illustrates eg the tradeoff between
  % full/tracklet formats depending on the sparsity of tracking.
  %
  % 
  % *Use of TrkFile at Runtime*
  % At runtime there is often a need for "results with visualization" which
  % is currently implemented with TrkFile+TV where TV is a 
  % TrackingVisualizerMT or TrackingVisualizerTracklets. (TVTracklets
  % currently uses a ptrx as a historic impl detail; note that the ptrx and
  % TrkFile should share data arrays under the hood.) 
  %
  % When using TrkFiles for runtime purposes (eg: .labels2) it may be 
  % desirable to switch between tracklet and dense formats for performance
  % reasons. Going further, one can imagine a situation where a TrkFile 
  % alone is inadequate for runtime purposes (eg imagine needing both dense 
  % and tracklet formats simultaneously). If this were to occur it would
  % probably not make sense to continue runtime-related functionality to
  % TrkFile, and instead create a new class (possibly composite, holding a 
  % TrkFile etc). This situation has not occurred yet and for now TrkFile 
  % is being used for runtime applications.
  
  methods
    
    function obj = TrkFile(npts,itlts,startframes,endframes,trkfldsextra,...
        varargin)
      if nargin==0
        % Trkfile()
        return;
      elseif nargin==2
        % TrkFile(npts,itlts)
        ntgts = numel(itlts);
        startframes = ones(1,ntgts);
        endframes = zeros(1,ntgts);
        trkfldsextra = {'pTrkConf'};
        obj = TrkFile(npts,itlts,startframes,endframes,trkfldsextra);
      end
      
      assert(isequal(numel(itlts),numel(startframes),numel(endframes)));
  
      obj.npts = npts;
      
      nfs = endframes-startframes+1;
      obj.pTrk = arrayfun(@(x)nan(npts,2,x),nfs,'uni',0);
      nowdt = now;
      obj.pTrkTS = arrayfun(@(x)nowdt*ones(npts,x),nfs,'uni',0);
      obj.pTrkTag = arrayfun(@(x)false(npts,x),nfs,'uni',0);
      obj.trkfldsextra = {};
      s_ptrk_dim = obj.ptrk_fns_dimensional;
      flds_ptrk_dim = fieldnames(s_ptrk_dim);
      for f=trkfldsextra(:)',f=f{1}; %#ok<FXSET>
        if ~isprop(obj,f)
          obj.addprop(f);
          if any(strcmp(f,flds_ptrk_dim))
            dim = s_ptrk_dim.(f);
            obj.(f) = arrayfun(@(x)nan(npts,dim,x),nfs,'uni',0);
          else
            obj.(f) = arrayfun(@(x)nan(npts,x),nfs,'uni',0);
          end
          obj.trkfldsextra{end+1} = f;
        end
      end
      
      obj.pTrkiTgt = itlts;
      obj.startframes = startframes;
      obj.endframes = endframes;
      
      for i=1:2:numel(varargin)
        obj.(varargin{i}) = varargin{i+1};
      end
      
      obj.isfull = false;
    end

%     function add3dpts(obj)
%       nfs = obj.endframes-obj.startframes+1;
%       obj.pTrkSingleView = arrayfun(@(x)nan(obj.npts,2,x),nfs,'uni',0);
%     end
    
    function initFromArraysFull(obj,ptrk,varargin)
      % 
      % ptrk: must be supplied, to set .pTrk
      % varargin: p-v pairs for remaining props
      %
      % Example: tfObj = TrkFile(myTrkPos,'pTrkTS',myTrkTS);
      %
      % Note: init* methods do not clear old state that is already set
      
      assert(isnumeric(ptrk));

      obj.pTrk = ptrk;
      
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
      
      obj.isfull = true;
      obj.initUnsetFull();
    end
    
    function initUnsetFull(obj)
      % Set any "unset" props
      % Note: init* methods do not clear old state that is already set

      assert(obj.isfull);
      assert(~isempty(obj.pTrk));
      
      [npttrk,d,nfrm,ntgt] = size(obj.pTrk);
      if d~=2
        error('TrkFile:TrkFile','. Trkfile is primarily designed for 3. but Expected d==2.');
      end

      if isequal(obj.pTrkTS,TrkFile.unsetVal)
        obj.pTrkTS = zeros(npttrk,nfrm,ntgt);
      end
      validateattributes(obj.pTrkTS,{'numeric'},{'size' [npttrk nfrm ntgt]},'','pTrkTS');
      
      if isequal(obj.pTrkTag,TrkFile.unsetVal) || ...
          ( all(size(obj.pTrkTag)==[npttrk nfrm]) && iscell(obj.pTrkTag))
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
      
      obj.nframes = nfrm;
      
    end

    function initFromSparse(obj,s)

      flds = fieldnames(s);
      sz = size(s.p);
      npts = sz(1)/2;
      frm0 = double(min(s.frm));
      frm1 = double(max(s.frm));
      T = frm1-frm0+1;
      tgts = unique(double(s.tgt));
      ntgts = numel(tgts);

      pTrk = nan([sz(1),T,ntgts]);
      isocc = isfield(s,'occ');
      if isocc,
        pTrkTag = false([npts,T,ntgts]);
      end
      for itgt = 1:numel(tgts),
        tgt = tgts(itgt);
        idxcurr = s.tgt == tgt;
        pTrkcurr = nan([sz(1),T]);
        pTrkcurr(:,s.frm(idxcurr)-frm0+1) = s.p(:,idxcurr);
        pTrk(:,:,itgt) = pTrkcurr;
        if isocc,
          pTrkTagcurr = false([npts,T]);
          pTrkTagcurr(:,s.frm(idxcurr)-frm0+1) = s.occ(:,idxcurr)>0;
          pTrkTag(:,:,itgt) = pTrkTagcurr;
        end
      end
      obj.pTrk = reshape(pTrk,[npts,2,T,ntgts]);
      if isocc,
        obj.pTrkTag = pTrkTag;
      end
      obj.pTrkFrm = reshape(frm0:frm1,[T,1]);
      obj.pTrkiTgt = tgts;
      extraflds = setdiff(flds,{'p','ts','occ','frm','tgt'});
      if ~isempty(extraflds),
        ss = sprintf(' %s',extraflds{:});
        warning('Unknown extra fields ignored:%s',ss);
      end
      obj.startframes = frm0;
      obj.endframes = frm1;

      obj.isfull = true;
      obj.initUnsetFull();
      
    end
    
    function initFromTableFull(obj,s,view,movi,outfile,varargin)
      %
      % Note: init* methods do not clear old state that is already set
      % 
      % this API is strange s and the pvs

      assert(isstruct(s));
      if ~isfield(s, 'pred_locs') ,
        outfile  %#ok<NOPRT> 
        s  %#ok<NOPRT> 
        destination_path = fullfile(tempdir(), 'heisenbug_trk_file_missing_pred_locs.trk') ;
        [did_copy_succeed, msg] = copyfile(outfile, destination_path) ;
        if ~did_copy_succeed ,
          warning('Unable to copy a suspect trk file to the temp dir: %s', msg) ;
        end
        error('s lacks pred_locs field') ;
      end
      if ~isfield(s, 'toTrack') ,
        outfile  %#ok<NOPRT> 
        s  %#ok<NOPRT> 
        destination_path = fullfile(tempdir(), 'heisenbug_trk_file_missing_toTrack.trk') ;
        [did_copy_succeed, msg] = copyfile(outfile, destination_path) ;
        if ~did_copy_succeed ,
          warning('Unable to copy a suspect trk file to the temp dir: %s', msg) ;
        end
        error('s lacks toTrack field') ;
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
      
%       [movfiles,nviewstrack] = ...
%         TrkFile.convertJSONCellMatrix(s.movieFiles);
      % movfiles = s.movieFiles;

      pred_locs = s.pred_locs.locs;
      ndim = ndims(pred_locs);
      if ndim ==4
        isma = true;
      else
        isma = false;
      end

      npts = size(pred_locs,ndim-1);
      d = size(pred_locs,ndim);
      if isma
        nanimals = size(pred_locs,ndim-2);
      else
        nanimals = 1;
      end
      assert(d == 2);
      n = size(pred_locs,1)*nanimals;
      iTgt = nan(n,1);
      frm = nan(n,1);
      movidx = false(n,1);
      off = 0;
%       movi = find(strcmp(obj.movfile,movfiles));
%       assert(numel(movi)==1);      
      nexttgt = 1;
      pTrk = nan(n,npts,d);
      pTrkTS = ones(n,npts)*now;
      pTrkTag = nan(n,npts);
      ists = isfield(s,'pred_ts');
      istag = isfield(s.pred_locs,'occ');

      for i = 1:numel(s.toTrack)
        
        x = s.toTrack{i};
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

        if double(x{1}) == movi,
          if isma
            % all predictions are new animals
            for fndx = 1:nfrm
              for andx = 1:nanimals
                curndx = (off+fndx-1)*nanimals+andx;
                frm(curndx) = frm0+fndx-1;
                curp = permute(pred_locs(off+fndx,andx,:,:),[3,4,1,2]);
                if ~all(isnan(curp))
                  iTgt(curndx) = nexttgt;
                  nexttgt = nexttgt+1;
                  movidx(curndx) = true;
                  pTrk(curndx,:,:) = curp;
                  if ists
                    pTrkTS(curndx,:) = permute(s.pred_locs.pred_ts(off+fndx,andx,:),[2,3,1]);
                  end
                  if istag
                    pTrkTag(curndx,:) = permute(s.pred_locs.occ(off+fndx,andx,:),[2,3,1]);
                  end
                end
              end
            end
          else
            iTgt(off+1:off+nfrm) = double(x{2});
            movidx(off+1:off+nfrm) = true;
            frm(off+1:off+nfrm) = frm0:frm1;
            pTrk(end+1:end+nfrm,:,:) = pred_locs(off+1:off+nfrm,:,:);
            if ists
              pTrkTag(end+1:end+nfrm,:) = s.pred_locs.pred_ts(off+1:off+nfrm,:);
            end
            if istag
              pTrkTS(end+1:end+nfrm,:) = s.pred_locs.occ(off+1:off+nfrm,:);
            end
          end
        end
        off = off + nfrm;
      end
      % pTrk is [npttrked x 2 x nfrm x ntgt]
      % p = reshape(p,[npts*d nF nTgt]);
      % pcol = p(:,i,j);
      % pred_locs is n x npts x 2
      
      nidx = nnz(movidx);
      [obj.pTrkiTgt,~,tgtidx] = unique(iTgt(movidx));
      frm = frm(movidx);
      pTrk = pTrk(movidx,:,:);
      pTrkTS = pTrkTS(movidx,:);
      if istag
        pTrkTag = pTrkTag(movidx,:);
      end
      nfrm = max(frm);
      ntgt = numel(obj.pTrkiTgt);
      
      obj.pTrk = cell(1,ntgt);
      obj.pTrkTS = cell(1,ntgt);
      if istag
        obj.pTrkTag = cell(1,ntgt);
      end

      startframes = zeros(1,ntgt);
      endframes = zeros(1,ntgt);
      for i = 1:ntgt
        curfr = frm(tgtidx==i);
        startframes(i) = min(curfr);
        endframes(i) = max(curfr);
        nfr = endframes(i) - startframes(i)+1;
        obj.pTrk{i} = nan(npts,d,nfr);
        obj.pTrkTS{i} = nan(npts,nfr);
        if istag
          obj.pTrkTag{i} = false(npts,nfr);
        end
      end
      obj.startframes = startframes;
      obj.endframes = endframes;

      for i = 1:nidx
        fndx = frm(i)-startframes(tgtidx(i))+1;
        curt = tgtidx(i);
        obj.pTrk{curt}(:,:,fndx) = permute(pTrk(i,:,:),[2,3,1]);
        obj.pTrkTS{curt}(:,fndx) = permute(pTrkTS(i,:),[2,1]);
        if istag
          obj.pTrkTag{curt}(:,fndx) = permute(pTrkTag(i,:),[2,1])<0.5;
        end
      end
      
      obj.isfull = false;
    end  % function    
    
    function initFromTracklet(obj,s)
      flds = fieldnames(s);
      obj.trkfldsextra = {};
      for prop=flds(:)',prop=prop{1}; %#ok<FXSET>
        if ~isprop(obj,prop)
          %warningNoTrace('Adding TrkFile property ''%s''.',prop);
          obj.addprop(prop);
          obj.trkfldsextra{end+1} = prop;
        end
        obj.(prop) = s.(prop);
        if isnumeric(obj.(prop)) && isinteger(obj.(prop)),
          obj.(prop) = double(obj.(prop));
        end
      end
      
      flds = obj.trkflds();
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        v = obj.(f);
%         if strcmp(f,'pTrkSingleView') && ~TrkFile.has3Dpts(obj)
%           % mainly for pTrkSingleView because we don't want to set it
%           % unless the trkfile has it. The logic is not foolproof or rather completely tested. So if it
%           % breaks this should be updated. MK 20230515
%           continue
%         end
        if strcmp(obj.(f),TrkFile.unsetVal)
          continue;
        end
        if ~iscell(v)
          warning('%s is not a cell',f);
          continue;
        end
        tfTag = strcmp(f,'pTrkTag');
        for i=1:numel(v)
          if isempty(v{i}) && size(v{i},1)==0 
            % compensate for apparent bug on Py 
            % https://github.com/frejanordsiek/hdf5storage/issues/114
            %
            % Only do this when first dim has 0 len. trkfiles that have
            % been exported/saved will be fixed. the other dims should
            % not be zero so that leaves the numframes dim to have len(0)
            % for an empty array. The numframes dim should be last, but on
            % load from Py due to the hdf5storage it will be first.
            
            ndim = ndims(v{i});            
            v{i} = permute(v{i},ndim:-1:1);
          elseif tfTag && isnumeric(v{i}) && ~isempty(v{i}) 
            v{i}(isnan(v{i})) = 0;
            v{i} = logical(v{i});
          end
        end
        obj.(f) = v;
      end
      
      % semi-tmp. backend producing inconsistent trkfiles
      if numel(obj.pTrkiTgt)~=numel(obj.pTrk)
        warningNoTrace('Unexpected/corrupt .pTrkiTgt field in TrkFile.');
        obj.pTrkiTgt = 1:numel(obj.pTrk);
      end
      
      if ~isempty(obj.pTrk)
        obj.npts = size(obj.pTrk{1},1); 
        obj.nframes = double(max(obj.endframes));
      else
        try 
          obj.npts = obj.trkInfo.params.n_classes;
        catch ME
          warningNoTrace('Could not determine .npts from tracklet TrkFile.');
          obj.npts = nan;
        end
      end
      obj.isfull = false;
    end
    
    function clearTracklet(obj)
      assert(~obj.isfull);
      
      for i=1:obj.ntracklets
        obj.startframes(i) = 0;
        obj.endframes(i) = -1;
      end
      
      tflds = obj.trkflds();
      for f=tflds(:)',f=f{1}; %#ok<FXSET>
        for i=1:obj.ntracklets
          v = obj.(f){i};
          if ndims(v)==3
            obj.(f){i} = v(:,:,1:0);
          else
            obj.(f){i} = v(:,1:0);
          end
        end
      end      
    end
    
    function rmTracklet(obj,iTlt)
      assert(~obj.isfull);
      
      tf = numel(obj.pTrkiTgt)==obj.ntracklets;
      if tf
        obj.pTrkiTgt(:,iTlt) = [];
      end
      
      obj.startframes(:,iTlt) = [];
      obj.endframes(:,iTlt) = [];
      
      tflds = obj.trkflds();
      for f=tflds(:)',f=f{1}; %#ok<FXSET>
        assert(isrow(obj.(f)));
        obj.(f)(:,iTlt) = [];
      end
    end
    
    function s = structizePrepSave(obj)
      warnst = warning('off','MATLAB:structOnObject');
      s = struct(obj);
      warning(warnst);
      
      mc = ? TrkFile;
      mprops = mc.PropertyList;
      tfrm = [mprops.Dependent] | [mprops.Constant];
      rmpropnames = {mprops(tfrm).Name}';
      s = rmfield(s,rmpropnames);
      
      for f=fieldnames(s)',f=f{1}; %#ok<FXSET>
        if ischar(s.(f)) && strcmp(s.(f),TrkFile.unsetVal)
          s = rmfield(s,f);
        end
      end
    end

    function save(obj,filename)
      % Saves to filename; ALWAYS OVERWRITES!!
      
      s = obj.structizePrepSave();
      save(filename,'-mat','-struct','s');
    end
    
  end
  
  methods (Static) % load
    
    % v = isValidLoadFullMatrix(s)
    % whether the struct resulting from loading is a full matrix trk file
    function v = isValidLoadFullMatrix(s)
      v = isfield(s,'pTrk');
    end
    function v = isValidSparse(s)
      v = all(isfield(s,{'p','frm','tgt'}));
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
      elseif TrkFile.isValidSparse(s),
        filetype = 'sparse';
      else
        filetype = '';
      end
    end
    
    function trkfileObj = load(filename,varargin)

      [movfile,issilent,movnframes] = myparse(varargin,...
        'movfile','',...
        'issilent',false,...
        'movnframes',[] ...
        );
      
      ntries = 5;
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
      if isempty(filetype),
        error('TrkFile:load',...
          'File ''%s'' is not a valid saved trkfile structure.',filename);
      end
      
%       if strcmp(filetype,'tracklet')
%         %%% Tracklet early return %%%
%         trkfileObj = load_tracklet(s);
%         % We do this right here, upon entry into APT, but this might more
%         % properly be done further inside the App (eg at vizInit-time) as 
%         % .x, .y are more for viz purposes.
%         trkfileObj = TrxUtil.ptrxAddXY(trkfileObj); 
%         [trkfileObj.movfile] = deal(movfile);
%         return;        
%       end        
        
      s = TrkFile.modernizeStruct(s);      
      trkfileObj = TrkFile();
      switch filetype,
        case 'tracklet'
          s.movfile = movfile;
          trkfileObj.initFromTracklet(s);
          if ~isempty(movnframes)
            trkfileObj.initFrm2Tlt(movnframes);
          end
        case {'fullmatrix' 'table'}
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

          if strcmp(filetype,'fullmatrix')
            pTrk = s.pTrk;
            pvs = struct2pvs(rmfield(srecog,'pTrk'));
            trkfileObj.initFromArraysFull(pTrk,'movfile',movfile,pvs{:});
          else % table       
            pTrk = struct;
            pTrk.pred_locs = s.pred_locs;
            pTrk.to_track = s.to_track;
            pvs = struct2pvs(srecog);
            trkfileObj.initFromTableFull(pTrk,'movfile',movfile,pvs{:});  % This seems wrong, but I'm not sure how to fix -- ALT, 2024-12-19
          end
          
          if issilent,
            fnsunrecog = setdiff(fnsunrecog,TrkFile.listfile_fns);
            for f=fnsunrecog(:)',f=f{1}; %#ok<FXSET>
              trkfileObj.addprop(f);
              trkfileObj.(f) = s.(f);
            end
          end
        case 'sparse',
          trkfileObj.initFromSparse(s);
        otherwise ,
          error('Internal error: filetype is not a known value') ;
      end  % switch
    end  % function
    
%     function trkfileObj = loadsilent(filename,varargin)
%       trkfileObj = TrkFile.load(filename,'issilent',true,varargin{:});
%     end
    
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
    end
    
  end
  
  methods % tracklet
    
    function trkflds = trkflds(obj)
      % tracklet 'pTrk' fields (cell contents etc)
      
      fns = fieldnames(obj);
      trkflds = fns(startsWith(fns,'pTrk'));
      trkflds = setdiff(trkflds,{'pTrkiPt' 'pTrkFrm' 'pTrkiTgt' ...
        'pTrkFull' 'pTrkFullFT' 'pTrk3d'});
    end
        
    function objMerged = merge(obj,other_objs,varargin)
      % objMerged = merge(obj0,obj1,obj2,...)
      %
      % returns new TrkFile with merge contents. tracklet-style merge.
      %
      % This method uses .pTrkiTgt as a "primary key" when merging
      % tracklets, ie tracklets with the same value of pTrkiTgt are merged
      % together, while tracklets with differing pTrkiTgt are simply
      % appended/concatenated.
      % 
      % This method could operate on a vector of TrkFile objs, but 
      % MATLAB behavior as of 2020b on
      % vectors-of-handles-with-dynamic-props can be a little subtle. Eg,
      % fieldnames(scalarObj) includes dynamicprops in the output, but
      % fieldnames(vectorOfObjs) does not. So for simplicitly, provide
      % scalar TrkFileObjs as distinct args.
      
      assert(~obj.isfull,'Attempt to merge non-tracklets.');

      [new_ids] = myparse(varargin,'new_ids',false);
      
      % step 1: get all unique tgts; get all their start/endframes      
      allobjs = [{obj}; other_objs];
      
      nobj = numel(allobjs);
      tgt_starts = zeros(1,nobj);
      if new_ids
        itgtsAll = cell(1,nobj);
        cur_s = 0;
        for ndx = 1:nobj
          tgt_starts(ndx) = cur_s;
          itgtsAll{ndx} = allobjs{ndx}.pTrkiTgt(:) + cur_s;
          cur_s = cur_s + numel(allobjs{ndx}.pTrkiTgt);
        end
      else
        itgtsAll = cellfun(@(x)x.pTrkiTgt(:),allobjs,'uni',0);
      end
      %sfsAll = {allobjs.startframes};
      %efsAll = {allobjs.endframes};
      
      itgtsun = unique(cat(1,itgtsAll{:}));
      itgtmax = max(itgtsun);
      cls = class(obj.startframes);
      itgt2spep = [intmax('int64')*ones(1,itgtmax,'int64');...
                   intmin('int64')*ones(1,itgtmax,'int64')]; % rows: sf, ef
      itgt2spep = eval(sprintf('%s(itgt2spep)',cls));
      for iobj=1:nobj
        o = allobjs{iobj};
        itgtsI = o.pTrkiTgt+tgt_starts(iobj);
        sfsI = o.startframes;
        efsI = o.endframes;
        tfhasres = sfsI<=efsI;
        itgt2spep(1,itgtsI(tfhasres)) = min(itgt2spep(1,itgtsI(tfhasres)),sfsI(tfhasres));
        itgt2spep(2,itgtsI(tfhasres)) = max(itgt2spep(2,itgtsI(tfhasres)),efsI(tfhasres));
      end
      
      sfsNew = itgt2spep(1,itgtsun);
      efsNew = itgt2spep(2,itgtsun);
      tfNoFrms = sfsNew>efsNew;
      sfsNew(tfNoFrms) = 0;
      efsNew(tfNoFrms) = -1;
      
      % 2. initialize new TrkFile with empty trkflds of right size
      % (nan-filled)

      objMerged = TrkFile(obj.npts,itgtsun,sfsNew,efsNew,obj.trkflds);
%       if TrkFile.has3Dpts(obj)
%         objMerged.add3dpts();
%       end
     
      % 3. init logical "isset" flag [nrms] indiciating whether frames set
      nfsNew = efsNew-sfsNew+1;      
      frmsAreSet = arrayfun(@(x)false(x,1),nfsNew,'uni',0);
      % frmsAreSet{jall} is nframes_jall x 1 logical indicator vec covering
      % objMerged.startframes(jall):objMerged.endframes(jall)
      
      s_ptrk_dim = obj.ptrk_fns_dimensional;
      flds_ptrk_dim = fieldnames(s_ptrk_dim);
      
      % 4. for each trkfile, for each iTgt, overlay flds. warn if 
      % overlapping frmsAreSet
      itgt2jall = nan(itgtmax,1);
      itgt2jall(objMerged.pTrkiTgt) = (1:numel(objMerged.pTrkiTgt))';
      % jall is index into objMerged.pTrk{jall} etc
      for iobj=1:nobj
        o = allobjs{iobj};
        trkfldso = o.trkflds();
        ntgt = numel(o.pTrkiTgt);
        for j=1:ntgt
          itgt = o.pTrkiTgt(j)+tgt_starts(iobj);
          sp = o.startframes(j);
          ep = o.endframes(j);
          if sp<=ep
            %nf = ep-sp+1;
            %off = 1-sp;
            jall = itgt2jall(itgt);          
            spall = objMerged.startframes(jall);
            offall = 1-spall;
            idxall = sp+offall:ep+offall;
            noverlap = nnz(frmsAreSet{jall}(idxall));
            frmsAreSet{jall}(idxall) = true;
            if noverlap>0
              warningNoTrace('Target %d: %d frames covered by two trkfiles. Overwriting.',...
                itgt,noverlap);
            end

            % write trkflds
            for f=trkfldso(:)',f=f{1}; %#ok<FXSET>
              if ~isprop(objMerged,f)
                objMerged.(f) = cell(1,numel(objMerged.pTrkiTgt));
              end
%               if strcmp(f,'pTrkSingleView') && ~TrkFile.has3Dpts(o)
%                 continue;
%               end
              if isequal(o.(f),TrkFile.unsetVal),
                continue;
              end
              if any(strcmp(f,flds_ptrk_dim))
                objMerged.(f){jall}(:,:,idxall) = o.(f){j}; 
              else
                objMerged.(f){jall}(:,idxall) = o.(f){j};
              end
            end
          end
        end
      end
      
      % AL 20220621
      % Not doing anything with frmsAreSet for now. objMerged should be
      % initialized with nans for trkflds() props
          
      % not sure what to do here
      obj.pTrkFrm = [];
%       if ~isequal(obj.movfile,obj1.movfile)
%         warningNoTrace('Trkfiles have differing .movfile fields.');
%       end
    end
        
    function mergeMultiView(obj,varargin)
      % mergeMultiView(obj,obj2,obj3,...)
      %
      % *Updates obj in place*
      %
      % Note, using matlab.mixin.Copyable doesn't work well with TrkFile bc
      % dynamicprops are not copied.
      %
      % Assumes obj2, obj3, ... correspond to views 2, 3, ...
      %
      % For now this performs strict tests that all objs track the same 
      % iTgts, start/endframes, etc.
      
      assert(~obj.isfull);
      %FLDSSAME = {'pTrkiPt' 'pTrkFrm' 'pTrkiTgt' 'frm2tlt'};
      FLDSSAME = {'pTrkiPt' 'pTrkFrm' 'pTrkiTgt'};
      FLDSMERG = obj.trkflds();
      FLDSMERG = FLDSMERG(:)';
      
      nobj = numel(varargin);
      for iobj=1:nobj
        o2 = varargin{iobj};
        assert(~o2.isfull);
        for f=FLDSSAME,f=f{1}; %#ok<FXSET>
          assert(isequaln(obj.(f),o2.(f)),'Objects differ in field ''%s''.',f);
        end
        for f=FLDSMERG,f=f{1}; %#ok<FXSET>
          % cat along "npoints" dim
          obj.(f) = cellfun(@(x,y)cat(1,x,y),obj.(f),o2.(f),'uni',0);
        end
        obj.npts = obj.npts + o2.npts;
      end
    end
    
    function t = tableform(obj,varargin)
      [ftonly,labelsColnames,aliveonly] = myparse(varargin,...
        'ftonly',false, ...
        'labelsColNames',false, ... % if true, use columns compatable with Labels.fromtable()
        'aliveonly',false ... % whether to check that all returned frames are tracked
        );

      if ftonly && aliveonly,
        t = obj.getTrackedFrames();
        return;
      end
      assert(~obj.isfull);
      frm = arrayfun(@(x,y)(x:y)',obj.startframes,obj.endframes,'uni',0);
      iTgt = cellfun(@(x,y)repmat(x,numel(y),1),num2cell(obj.pTrkiTgt),frm,'uni',0);
      s = struct();
      s.frm = cat(1,frm{:});
      s.iTgt = cat(1,iTgt{:});
      
      flds = obj.trkflds();
      if aliveonly,
        sf = obj.getStartFrame();
        ef = obj.getEndFrame();
        isalive = cell(1,obj.ntracklets);
        for i = 1:obj.ntracklets,
          isalive{i} = obj.isalive(sf:ef,i);
        end
      end
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        v = obj.(f);
        nd = cellfun(@ndims,v);
        assert(all(nd==nd(1)));
        nd = nd(1);
        v = cellfun(@(x)permute(x,[nd 1:nd-1]),v,'uni',0); % put 'frame' dim first
        % convert to 2d arrays (in particular for pTrk)
        for i=1:numel(v)
          if aliveonly && isalive,
            v{i} = v{i}(isalive{i},:);
          else
            szv = size(v{i});
            v{i} = reshape(v{i},szv(1),prod(szv(2:end)));
          end
        end
        %AL 202202: this line prob should work but leads to incorrect sizes
        %for empty arrs
        %v = cellfun(@(x)reshape(x,size(x,1),[]),v,'uni',0); 
        s.(f) = cat(1,v{:});
      end
      
      t = struct2table(s);
      
      if labelsColnames
        COLSFROM = {'^pTrk$' '^pTrkTS$' '^pTrkTag$'};
        COLSTO = {'p' 'pTS' 'tfocc'};
        colnames = regexprep(t.Properties.VariableNames,COLSFROM,COLSTO);
        t.Properties.VariableNames = colnames;
      end
    end
    
    function ft = getTrackedFrames(obj)
      sf = obj.getStartFrame();
      ef = obj.getEndFrame();
      ft = table(nan(0,1),nan(0,1),'VariableNames',{'frm' 'iTgt'});
      for iTgt = 1:numel(sf),
        v = obj.isalive(sf(iTgt):ef(iTgt),iTgt);
        if ~any(v),
          continue;
        end
        ftcurr = struct;
        ftcurr.frm = find(v)+sf(iTgt)-1;
        ftcurr.iTgt = zeros(size(ftcurr.frm))+iTgt;
        ft = [ft;struct2table(ftcurr)]; %#ok<AGROW> 
      end

    end

    function v = isalive(obj,f,itgt)
      
      v = obj.getPTrkFT(f,itgt);
%       if obj.isfull,
%         % pTrk: [npttrked x 2 x nfrm x ntgt], like labeledpos
%         if nargin < 3 || isempty(itgt),
%           v = permute(any(any(~isnan(obj.pTrk(:,:,f,:)),1),2),[3,4,1,2]);
%         elseif isempty(f),
%           v = permute(any(any(~isnan(obj.pTrk(:,:,:,itgt)),1),2),[3,4,1,2]);
%         else
%           v = permute(any(any(~isnan(obj.pTrk(:,:,f,itgt)),1),2),[3,4,1,2]);
%         end
%         return;
%       end
%       
%       if nargin < 3 || isempty(itgt),
%         % all targets, v will be numel(f) x ntgts
%         
%         v = f(:) >= obj.startframes & f(:) <= obj.endframes;
%       elseif isempty(f),
%         % all frames, v will be nframes x numel(itgt)
%         v = false(obj.nframes,numel(itgt));
%         for i = 1:numel(itgt),
%           v(i,obj.startframes(itgt(i)):obj.endframes(itgt(i))) = true;
%         end
%       elseif obj.ntracklets==0
%         v = false;
%       else
%         % v will be numel(f) x numel(itgt)
%         v = f(:) >= obj.startframes(itgt) & f(:) <= obj.endframes(itgt);
%       end
      
    end
    
    function tf = hasdata(obj)
      if obj.isfull
        tf = any(~isnan(obj.pTrk(:)));
      else
        tf = any(obj.endframes>=obj.startframes);
      end  
    end
    
%     function v = frm2tltnnz(obj)
%       if obj.isfull,
%         v = nnz(any(any(~isnan(obj.pTrk(:,:,f,:)),1),2));
%       else
%         v = sum(obj.endframes-obj.startframes+1);
%       end
%     end
    
    function initFrm2Tlt(obj,nfrm)
      % Initialize .frm2tlt property; implicitly sets .T1 
      %
      % nfrm (opt). total number of frames, eg in movie. If not specified,
      % max(obj.endframes) is used.

      assert(~obj.isfull);
      
      if nargin < 2
        nfrm = max(obj.endframes);
      end
      
      obj.nframes = nfrm;
      
%       if size(obj.frm2tlt,1)==nfrm && ~isempty(obj.frm2tltnnz)
%         warningNoTrace('frm2tlt maybe already initted.');
%       end
%       
%       ntgt = numel(obj.pTrk);
%       f2t = false(nfrm,ntgt);
%       sf = obj.startframes;
%       ef = obj.endframes;
%       for j=1:ntgt
%         f2t(sf(j):ef(j),j) = true;
%       end
%       obj.frm2tlt = f2t;
%       obj.frm2tltnnz = nnz(f2t); % number of live (frm,tlt) pairs
    end
    
    function [n,f0,f1] = maxTrackletsLive(obj)
      % maximum number of tracklets live at a given moment
      tfkeep = obj.startframes<obj.endframes;      
      f0 = min(obj.startframes(tfkeep));
      f1 = max(obj.endframes(tfkeep));
      cnt = zeros(f1-f0+1,1);
      for i=1:obj.ntracklets
        idx = (obj.startframes(i):obj.endframes(i)) - f0 + 1;
        cnt(idx) = cnt(idx)+1;
      end
      n = max(cnt);
    end  

    function sf = getStartFrame(obj,tgts)
      if nargin < 2,
        tgts = 1:obj.ntracklets;
      end
      if obj.isfull,
        warning('This code has not been debugged');
        sf = nan(size(tgts));
        for i = 1:numel(tgts),
          sfcurr = find(~isnan(obj.pTrk(1,1,:,tgts(i))),1);
          if ~isempty(sfcurr),
            sf(i) = sfcurr;
          end
        end
      else
        sf = reshape(obj.startframes(tgts),size(tgts));
      end
    end
    
    function ef = getEndFrame(obj,tgts)
      if nargin < 2,
        tgts = 1:obj.ntracklets;
      end
      if obj.isfull,
        warning('This code has not been debugged');
        ef = nan(size(tgts));
        for i = 1:numel(tgts),
          efcurr = find(~isnan(obj.pTrk(1,1,:,tgts(i))),1,'last');
          if ~isempty(efcurr),
            ef(i) = efcurr;
          end
        end
      else
        ef = reshape(obj.endframes(tgts),size(tgts));
      end
    end
    

    % TODO: consider API that excludes ~tfhaspred vals
    function [tfhaspred,xy,tfocc] = getPTrkFrame(obj,f,varargin)
      % get tracking for particular frame
      %
      % f: (absolute) frame index
      %
      % tfhaspred: [ntgt] logical vec, whether pred is present
      % xy: [npt x 2 x nfr x ntgt]
      % tfocc: [npt x nfr x ntgt]
      % KB 20220728 modified to take in any number of frames
      % optional arguments:
      % 'collapse': Whether to assume that f is a scalar, and collapse the
      % output to be [npt x 2 x ntgt], etc. Default: false
      
      % don't make isalive depend on 
      %tfhaspred = obj.isalive(f).'; 
      [tfhaspred,xy,tfocc] = obj.getPTrkFT(f,1:obj.ntracklets,varargin{:});
% 
%       nfrm = numel(f);
% 
%       if obj.isfull
%         xy = obj.pTrk(:,:,f,:);
%         tfocc = obj.pTrkTag(:,f,:);
%       else
%         %itgtsLive = find(tfhaspred);
%         npt = obj.npts;
%         ntgt = obj.ntracklets;
%         xy = nan(npt,2,nfrm,ntgt);
%         tfocc = false(npt,nfrm,ntgt);
%         pcell = obj.pTrk;
%         pcelltag = obj.pTrkTag;
%         offs = 1-obj.startframes;
% 
%   %       % 20210706 in obscure cases, eg lObj.currFrame can become a
%   %       % uint value. Now prohibited generally. The offsets are
%   %       % intXX's which causes an error in the addition below
%   %       offs = double(offs);
%   %       f = double(f);
% 
%         for j=1:ntgt,
%           ptgt = pcell{j};
%           ptag = pcelltag{j};
%           idx = f + offs(j);
%           xy(:,:,j) = ptgt(:,:,idx);
%           tfocc(:,j) = ptag(:,idx);
%         end
%       end
%       
%       tfhaspred = TrkFile.isAliveHelper(xy);

    end
    
    function permuteIds(obj,newids)
      
      assert(numel(unique(newids))==obj.ntracklets);
      
      if obj.isfull,
        if ~isequal(obj.pTrk,TrkFile.unsetVal),
          obj.pTrk = obj.pTrk(:,:,:,newids);
        end
        if ~isequal(obj.pTrkTS,TrkFile.unsetVal),
          obj.pTrkTS = obj.pTrkTS(:,:,newids);
        end
        if ~isequal(obj.pTrkTag,TrkFile.unsetVal),
          obj.pTrkTag = obj.pTrkTag(:,:,newids);
        end
      else
        if ~isequal(obj.pTrk,TrkFile.unsetVal),
          obj.pTrk = obj.pTrk(newids);
        end
        if ~isequal(obj.pTrkTS,TrkFile.unsetVal),
          obj.pTrkTS = obj.pTrkTS(newids);
        end
        if ~isequal(obj.pTrkTag,TrkFile.unsetVal),
          obj.pTrkTag = obj.pTrkTag(newids);
        end
        obj.startframes = obj.startframes(newids);
        obj.endframes = obj.endframes(newids);
        if isprop(obj,'pTrkConf'),
          obj.pTrkConf = obj.pTrkConf(newids);
        end
      end
%       if ~isequal(obj.pTrkiTgt,TrkFile.unsetVal),
%         obj.pTrkiTgt = obj.pTrkiTgt(newids);
%       end
      
    end
    
    function [tfhaspred,xy,tfocc,aux] = getPTrkFT(obj,f,iTgt,varargin)
      % get tracking for particular frame, tgt
      %
      % f: (absolute) frame index
      % iTgt: tracklet idx
      % if both f and iTgt are non-scalar, then the we will return
      % tracking info for the cross product of frame and target
      %
      % tfhaspred: [numel(fr) x numel(iTgt)] whether pred is present
      % xy: [npt x 2 x numel(fr) x numel(iTgt)]
      % tfocc: [npt x numel(fr) x numel(iTgt)]
      % aux (opt): [npt x numel(fr) x numel(iTgt) x numaux] Auxiliary
      %   stats, returned if 'auxflds' specified
      %
      % KB 20220728: modified to be able to give multiple frames and
      % targets. 
      % Optional arguments:
      % 'collapse': Whether to assume that f is a scalar, and collapse the
      % output to be [npt x 2 x ntgt], etc. Default: false

      % don't make this depend on isalive, let isalive call this
      %tfhaspred = obj.isalive(f,iTgt);

      [collapse,auxflds] = myparse(varargin,...
        'collapse',false,...
        'auxflds',[] ... % (opt). cellstr, aux fields to return. no checks are done that these fields exist
        );

      nfrm = numel(f);
      ntgt = numel(iTgt);
      npt = obj.npts;
      
      tfaux = ~isequal(auxflds,[]);
      if tfaux
        naux = numel(auxflds);
        aux = nan(npt,nfrm,ntgt,naux);
      else
        naux = 0;
        aux = [];
      end

      if obj.isfull
        if ~isempty(obj.startframes),
          ifrm = f - obj.startframes(1) + 1;
        else
          ifrm = f;
        end
        xy = obj.pTrk(:,:,ifrm,iTgt);
        tfocc = obj.pTrkTag(:,ifrm,iTgt);
        for iaux=1:naux
          fld = auxflds{iaux};
          aux(:,:,:,iaux) = obj.(fld)(:,ifrm,iTgt);
        end
      else
        xy = nan(npt,2,nfrm,ntgt);
        tfocc = false(npt,nfrm,ntgt); 
        pcell = obj.pTrk;
        pcelltag = obj.pTrkTag;
        for iaux=naux:-1:1
          %pcellaux only created if naux>=1
          pcellaux{iaux} = obj.(auxflds{iaux});
        end
        offs = 1-obj.startframes;
        sf = obj.startframes(iTgt);
        ef = obj.endframes(iTgt);

  %       % 20210706 in obscure cases, eg lObj.currFrame can become a
  %       % uint value. Now prohibited generally. The offsets are
  %       % intXX's which causes an error in the addition below
  %       offs = double(offs);
  %       f = double(f);

        for i=1:ntgt,
          isinterval = f >= sf(i) & f <= ef(i);
          if ~any(isinterval), continue; end

          j = iTgt(i);
          ptgt = pcell{j};
          ptag = pcelltag{j};

          idx = f(isinterval) + offs(j);
          xy(:,:,isinterval,i) = ptgt(:,:,idx);
          tfocc(:,isinterval,i) = ptag(:,idx);
          for iaux=1:naux
            paux = pcellaux{iaux}{j};
            aux(:,isinterval,i,iaux) = paux(:,idx);
          end
        end
      end

      tfhaspred = TrkFile.isAliveHelper(xy);

      if collapse,
        assert(numel(f) == 1);
        tfhaspred = permute(tfhaspred,[2,1]);
        xy = permute(xy,[1,2,4,3]);
        tfocc = permute(tfocc,[1,3,2]);
        if tfaux
          aux = permute(aux,[1,3,4,2]);
        end
      end

    end
    
    function p = getPTrkFullOrdered(obj)
      % return full pTrk array, ordered by x-coordinate of pt 1
      
      [maxtlt,f0,f1] = obj.maxTrackletsLive();
      nf = f1-f0+1;
      p = nan(obj.npts,2,nf,maxtlt);
      for f=f0:f1
        [tfhaspred,xy] = obj.getPTrkFrame(f);
        xy = xy(:,:,tfhaspred); % [npt x 2 x ntgtlive_f]
        ntgtlive = size(xy,3);
        if ntgtlive>0
          [~,idx] = sort(xy(1,1,:));
          xy = xy(:,:,idx);
          p(:,:,f-f0+1,1:ntgtlive) = xy;
        end         
      end
    end
    
    function frms = isLabeledT(obj,iTlt)
      % Pass iTlt==nan <=> "any target"
      
      if isnan(iTlt)
        v = false(obj.nframes,1);
        ntgt = numel(obj.startframes);
        for i=1:ntgt
          v(obj.startframes(i):obj.endframes(i)) = true;
        end
        frms = find(v);
      else
        frms = obj.startframes(iTlt):obj.endframes(iTlt);
      end
    end
    
%     function [xy,occ] = getPTrkTgtPadded(obj,iTlt,nfrmtot)
%       % Wrapper for getPTrkTgt which returns "full" timeseries of length
%       % nfrmtot
% 
%       [xy0,occ0,fr0] = obj.getPTrkTgt(iTlt);
%       if isnan(nfrmtot)
%         xy = xy0;
%         occ = occ0;
%         return;
%       end
% 
%       ntlt = numel(iTlt);
%       xy = nan(obj.npts,2,nfrmtot,ntlt);
%       occ = false(obj.npts,nfrmtot,ntlt);
%       if isempty(fr0)
%         % none
%       else        
%         xy(:,:,fr0,:) = xy0;
%         occ(:,fr0,:) = occ0;
%       end
%     end 
  
  function [tfhasdata,xy,occ,sf,ef,aux] = getPTrkTgt2(obj,iTlt,varargin)
    % Convenience wrapper
    [xy,occ,fr,aux] = obj.getPTrkTgt(iTlt,varargin{:});
    tfhasdata = ~isempty(xy); % note, xy could be all nans etc && TrkFile.isAliveHelper(xy);
    sf = fr(1);
    ef = fr(end);
  end

  function [xy,occ,fr,aux] = getPTrkTgt(obj,iTlt,varargin)
      % get tracking for particular target
      %
      % iTlt: tracklet index
      %
      % xy: [npt x 2 x numfrm x numel(iTlt)]. numfrm = ef-sf+1
      % occ: [npt x numfrm x numel(iTlt)]
      % fr: [numfrm] frames, labels 3rd dim of xy
      % aux: [npt x numfrm x numel(iTlt) x numaux] Auxiliary
      %   stats, meaningful if 'auxflds' specified
      
      auxflds = myparse(varargin,...
        'auxflds',[] ... % cellstr; addnl stats to return
        );
      
      tfhasdata = false;

      if any(iTlt > obj.ntracklets)
        warningNoTrace('Tracklet index exceeds available data.');
      elseif obj.isfull && ~isempty(obj.pTrk)
        tfhasdata = true;
        if ~isempty(obj.startframes),
          sf = obj.startframes(1);
        else
          sf = 1;
        end
        ef = size(obj.pTrk,3) + sf - 1;
      elseif ~obj.isfull && obj.hasdata()
        tfhasdata = true;
        % .startframes can contain 0s when tracklet has no data
        sf = max(min(obj.startframes),1);       
        ef = max(obj.endframes);
      end

      if ~tfhasdata
        % as if nframes=0
        ntlt = numel(iTlt);
        xy = nan(obj.npts,2,0,ntlt);
        occ = false(obj.npts,0,ntlt);
        fr = nan;
%         sf = nan;
%         ef = nan;

        tfAux = ~isequal(auxflds,[]);
        if tfAux
          naux = numel(auxflds);
          aux = nan(obj.npts,0,ntlt,naux);
        else
          aux = [];
        end
      else
        fr = sf:ef;
        [~,xy,occ,aux] = obj.getPTrkFT(fr,iTlt,'auxflds',auxflds);
        % first output arg (tfhaspred) of getPTrkFT is not used here. Note 
        % tfhasdata as returned by current function differs semantically 
        % from tfhaspred
        %
        % sf, ef already set
      end
    end
    
%     function xyaux = getPAuxTgt(obj,iTlt,ptrkfld,varargin)
%       % get aux tracking timeseries (eg confidences) for particular tgt
%       %
%       % iTlt: tracklet index
%       % ptrkfld: field, eg 'pTrkConf'
%       %
%       % xyaux: [npt x nfrm] auxiliary tracking
%       
%       missingok = myparse(varargin,...
%         'missingok',false ...
%         );
%       
%       npts = size(obj.pTrk{iTlt},1);
%       nfrm = obj.nframes;
%       %nfrm = size(obj.frm2tlt,1);
%       xyaux = nan(npts,nfrm);
%       
%       if isprop(obj,ptrkfld)      
%         pauxI = obj.(ptrkfld){iTlt};
%         f0 = obj.startframes(iTlt);
%         f1 = obj.endframes(iTlt);
%         xyaux(:,f0:f1) = pauxI;
%       elseif missingok
%         % none; xyaux all nans
%       else
%         error('Unknown field ''%s''.',ptrkfld);
%       end
%     end
    
    function trackletViz(obj,ax,varargin)
      plotargs = myparse(varargin,...
        'plotargs',{'linewidth',2} ...
        );
      
      axes(ax);
      hold on;
      %ntrx = numel(obj.pTrk);
      for i=1:obj.ntracklets
        t = obj.startframes(i):obj.endframes(i);
        p1 = squeeze(obj.pTrk{i}(1,1,:));
        plot(t,p1,plotargs{:});
      end
      grid on;
    end
  end

  methods (Static)
    function v = isAliveHelper(xy)
      nd = ndims(xy);      
      v = permute(any(any(~isnan(xy),1),2),[3:nd,1,2]);
    end

    function t = mergetablesMultiview(varargin)
      % t = mergetables(t1,t2,t3,...)
      %
      % Merge outputs of tableforms, assuming t1,t2,... represent
      % multiview data.
      %
      % Note the merge requirements here are not as strict as in
      % mergeMultiview.
      
      t0 = varargin{1};
      ntbl = numel(varargin); % assumed to be same as nviews
      
      colsall = t0.Properties.VariableNames;
      assert(isequal(colsall(1:2),{'frm' 'iTgt'}));
      colstrk = colsall(3:end);
      
      for i=2:ntbl
        t1 = varargin{i};
        t0 = outerjoin(t0,t1,'keys',{'frm' 'iTgt'},'mergekeys',true);
        for c=colstrk,c=c{1}; %#ok<FXSET>
          c0 = [c '_t0'];
          c1 = [c '_t1'];
          t0.(c) = [t0.(c0) t0.(c1)];          
        end
        t0 = t0(:,colsall);
      end
      
      if any(strcmp(colstrk,'pTrk'))  % might not be true eg for FT-only tables
        % pTrk now has col order [x1v1 x2v1 .. xkv1 y1v1 .. ykv1 x1v2 ...]
        % fix this up to standard [ <all x> <all y> ]
        n = height(t0);
        npt = size(t0.pTrk,2)/2/ntbl;
        ptrk = reshape(t0.pTrk,n,npt,2,ntbl); 
        ptrk = permute(ptrk,[1 2 4 3]);
        ptrk = reshape(ptrk,n,[]);
        t0.pTrk = ptrk;
      end
      
      t = t0;
    end
%     function res = has3Dpts(trk)
%       res = ~(ischar(trk.pTrkSingleView) && strcmp(trk.pTrkSingleView,TrkFile.unsetVal));
%     end

  end

  
  methods % table/full utils
    
    function obj2 = toTrackletFull(obj,varargin)
      % Return a new TrkFile which is the tracklet-form of obj
      
      compactify = myparse(varargin,...
        'compactify',true ... % if true, remove "empty"/nan frames from tracklets
        );
      
      assert(obj.isfull)
      
      [npts,~,nfrm,ntlt] = size(obj.pTrk);
      assert(isequal(obj.pTrkFrm,obj.pTrkFrm(1):obj.pTrkFrm(end)));
      sfs = repmat(obj.pTrkFrm(1),1,ntlt);
      efs = repmat(obj.pTrkFrm(end),1,ntlt);
      trkflds = obj.trkflds();
      obj2 = TrkFile(npts,obj.pTrkiTgt,sfs,efs,trkflds);      
      
      s_ptrk_dim = obj.ptrk_fns_dimensional;
      flds_ptrk_dim = fieldnames(s_ptrk_dim);

      for itlt=1:ntlt
        if compactify
          p = obj.pTrk(:,:,:,itlt);
          p = reshape(p,[],nfrm);
          tflivep = any(~isnan(p),1);          
          occ = obj.pTrkTag(:,:,itlt);
          tfliveocc = any(occ,1);
          tflive = tflivep | tfliveocc;
          idx0 = find(tflive,1,'first');
          idx1 = find(tflive,1,'last');
          if isempty(idx0)
            idx0 = 1;
            idx1 = 0;
          end
        else
          idx0 = 1;
          idx1 = nfrm;
        end
        
        idx = idx0:idx1; % index into .p* fields
        sfI = sfs(itlt) + idx0 - 1;
        efI = sfs(itlt) + idx1 - 1;
        
        for f=trkflds(:)',f=f{1}; %#ok<FXSET>
          v = obj.(f);
          if any(strcmp(f,flds_ptrk_dim))
            obj2.(f){itlt} = v(:,:,idx,itlt);
          else
            obj2.(f){itlt} = v(:,idx,itlt);
          end
        end
        obj2.startframes(itlt) = sfI;
        obj2.endframes(itlt) = efI;
      end
      
      obj2.isfull = false;
      obj2.npts = npts;      
    end
    
    % only one callsite and seems unnec
    function tbl = tableformFull(obj)
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
    
    % wont have callsites
    function mergePartialFull(obj1,obj2)
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
        obj1.hlpMergePartialFullTgts(obj2,flds{fldi},szs.(flds{fldi}),allidx,allnewidx,allitgt);
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
    
    function hlpMergePartialFullTgts(obj1,obj2,fld,valsz,allidx,allnewidx,allitgt)
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
    
%     function hlpMergePartial(obj1,obj2,fld,valsz,locfrm1,loctgt1,locfrm2,loctgt2)
%       % helper for mergePartial to handle additional/dynamic props
%       %
%       % mutates obj1.(fld)
%       % obsolete? 
%       
%       isprop1 = isprop(obj1,fld);
%       isprop2 = isprop(obj2,fld);
%       
%       if isprop1 || isprop2
%         val = nan(valsz);
%         valndim = numel(valsz);
%         if isprop1
%           switch valndim
%             case 3
%               val(:,locfrm1,loctgt1) = obj1.(fld);
%             case 4
%               val(:,:,locfrm1,loctgt1) = obj1.(fld);
%             otherwise
%               assert(false);
%           end
%         end
%         
%         if isprop2
%           switch valndim
%             case 3
%               val(:,locfrm2,loctgt2) = obj2.(fld);
%             case 4              
%               val(:,:,locfrm2,loctgt2) = obj2.(fld);
%             otherwise
%               assert(false);
%           end
%         end
%         
%         if ~isprop1
%           obj1.addprop(fld);
%         end        
%         obj1.(fld) = val;
%       end
%     end
    
    function indexInPlaceFull(obj,ipts,ifrms,itgts)
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
  end  % methods
  
  methods (Static)
    
    function [nFramesTracked,didload] = getNFramesTrackedTextFile(tfile)      
      didload = false;
      s = readtxtfile(tfile);
      nFramesTracked = TrkFile.getNFramesTrackedString(s);
    end

    function nFramesTracked = getNFramesTrackedString(s)      
      nFramesTracked = 0;
      PAT = '(?<numfrmstrked>[0-9]+)';
      toks = regexp(s,PAT,'names','once');
      if isempty(toks),
        return;
      end
      nFramesTracked = str2double(toks{1}.numfrmstrked);
    end

    function [nFramesTracked, didload] = getNFramesTrackedMatFile(tfile, varargin)
      
      nFramesTracked = 0;
      didload = false;
      ntries = 5;
      
      for tryi = 1:ntries,
        m = matfile(tfile);
        fns = fieldnames(m);

        if ismember('startframes',fns)
          if isempty(m.endframes)
            nFramesTracked = 0;
          else
            nFramesTracked = apt.totalFrameCountFromIntervals(m.startframes, m.endframes) ;
          end
          didload = true;
        % Making sure all of these are correct would be work, and not clear
        % if they're even needed these days.  So leave them commented and see if there
        % are any issues.  -- ALT, 2025-05-12
        % elseif ismember('pTrkFrm',fns)
        %   nFramesTracked = numel(m.pTrkFrm);
        %   didload = true;
        % elseif ismember('pTrk',fns),
        %   nd = ndims(m.pTrk);
        %   if nd == 3,
        %     nFramesTracked = nnz(~isnan(m.pTrk(1,1,:)));
        %   else
        %     nFramesTracked = nnz(~isnan(m.pTrk(1,1,:,:)));
        %   end
        %   didload = true;
        % elseif ismember('pred_locs',fns),
        %   nFramesTracked = nnz(~isnan(m.pred_locs(:,1)));
        %   didload = true;
        % elseif ismember('locs',fns)
        %   % gt mat-file
        %   % AL: not sure want nnz(~isnan(...)) here; what if a tracker
        %   % predicted occluded or something, could that mess stuff up?
        %   nFramesTracked = size(m.locs,1);
        %   didload = true;
        else
          didload = false;
          nFramesTracked = 0;
          if tryi > ntries/2,
            fprintf('try %d, variables in %s:\n',tryi,tfile);
            disp(m);
          end
          pause(5);
        end
        if didload,
          break
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
          [nFramesTracked,didload] = TrkFile.getNFramesTrackedTextFile(tfile);
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
