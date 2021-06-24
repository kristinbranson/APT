classdef TrkFile < dynamicprops
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
    listfile_fns = {'pred_locs','to_track','pred_tag','list_file',...
      'pred_ts','pred_conf'};
  end
  
  properties
    pTrk = TrkFile.unsetVal;     % [npttrked x 2 x nfrm x ntgt], like labeledpos
    pTrkTS = TrkFile.unsetVal;   % [npttrked x nfrm x ntgt], liked labeledposTS
    pTrkTag = TrkFile.unsetVal;  % [npttrked x nfrm x ntgt] logical, like labeledposTag
    pTrkiPt = TrkFile.unsetVal;  % [npttrked]. point indices labeling rows of .pTrk*. If 
                                 %  npttrked=labeled.nLabelPoints, then .pTrkiPt=1:npttrked.
    pTrkFrm = TrkFile.unsetVal;  % [nfrm]. frames tracked
    pTrkiTgt = TrkFile.unsetVal; % [ntgt]. targets (1-based Indices) tracked

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
  end
  properties (Dependent)
    ntlts
  end
  
  methods 
%     function v = get.npts(obj)
%       % assumes at least one tracklet...
%       v = size(obj.pTrk{1},1);
%     end
    function v = get.ntlts(obj)
      v = numel(obj.pTrk);
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
      for f=trkfldsextra(:)',f=f{1}; %#ok<FXSET>
        if ~isprop(obj,f)
          obj.addprop(f);
          obj.(f) = arrayfun(@(x)nan(npts,x),nfs,'uni',0);
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
      
      obj.nframes = nfrm;
      
    end
    
    function initFromTableFull(obj,s,varargin)
      %
      % Note: init* methods do not clear old state that is already set
      % 
      % this API is strange s and the pvs

      assert(isstruct(s));
      
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
      
      [movfiles,nviewstrack] = ...
        TrkFile.convertJSONCellMatrix(s.to_track.movieFiles);

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
      
      obj.isfull = true;
      obj.initUnsetFull();      
    end    
    
    function initFromTracklet(obj,s)
      flds = fieldnames(s);
      for prop=flds(:)',prop=prop{1}; %#ok<FXSET>
        if ~isprop(obj,prop)
          %warningNoTrace('Adding TrkFile property ''%s''.',prop);
          obj.addprop(prop);
        end
        obj.(prop) = s.(prop);
      end
      
      flds = obj.trkflds();
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        v = obj.(f);
        assert(iscell(v));
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
      
      obj.npts = size(obj.pTrk{1},1); 
      obj.isfull = false;
    end
    
    function clearTracklet(obj)
      assert(~obj.isfull);
      
      %tfInitFrm2Tlt = ~isequal(obj.frm2tlt,[]);
      %nfrm = size(obj.frm2tlt,1);
      
      for i=1:obj.ntlts
        obj.startframes(i) = 0;
        obj.endframes(i) = -1;
      end
      
      tflds = obj.trkflds();
      for f=tflds(:)',f=f{1}; %#ok<FXSET>
        for i=1:obj.ntlts
          v = obj.(f){i};
          if ndims(v)==3
            obj.(f){i} = v(:,:,1:0);
          else
            obj.(f){i} = v(:,1:0);
          end
        end
      end
      
%       if tfInitFrm2Tlt
%         obj.initFrm2Tlt(nfrm);
%       end
    end
    
    function s = structizePrepSave(obj)
      warnst = warning('off','MATLAB:structOnObject');
      s = struct(obj);
      warning(warnst);
      for f=fieldnames(s)',f=f{1}; %#ok<FXSET>
        if ischar(s.(f)) && strcmp(s.(f),TrkFile.unsetVal)
          s = rmfield(s,f);
        end
      end
      
      mc = ? TrkFile;
      mprops = mc.PropertyList;
      tfrm = [mprops.Dependent]; %  | [mprops.Hidden];
      rmpropnames = {mprops(tfrm).Name}';
      s = rmfield(s,rmpropnames);

      % above check removes .unsetVal prop!initFromTra
      
      s = rmfield(s,'listfile_fns');
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
            trkfileObj.initFromTableFull(pTrk,'movfile',movfile,pvs{:});
          end
          
          if issilent,
            fnsunrecog = setdiff(fnsunrecog,TrkFile.listfile_fns);
            for f=fnsunrecog(:)',f=f{1}; %#ok<FXSET>
              trkfileObj.addprop(f);
              trkfileObj.(f) = s.(f);
            end
          end
      end
    end
    
    function trkfileObj = loadsilent(filename,varargin)
      trkfileObj = TrkFile.load(filename,'issilent',true,varargin{:});
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
    end
    
  end
  
  methods % tracklet
    
    function trkflds = trkflds(obj)
      % tracklet 'pTrk' fields (cell contents etc)
      
      fns = fieldnames(obj);
      trkflds = fns(startsWith(fns,'pTrk'));
      trkflds = setdiff(trkflds,{'pTrkiPt' 'pTrkFrm' 'pTrkiTgt' ...
        'pTrkFull' 'pTrkFullFT'});
    end
        
    function objMerged = merge(obj,varargin)
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
      
      % step 1: get all unique tgts; get all their start/endframes      
      allobjs = [{obj} varargin];
      
      nobj = numel(allobjs);
      itgtsAll = cellfun(@(x)x.pTrkiTgt,allobjs,'uni',0);
      %sfsAll = {allobjs.startframes};
      %efsAll = {allobjs.endframes};
      
      itgtsun = unique(cat(2,itgtsAll{:}));
      itgtmax = max(itgtsun);
      cls = class(obj.startframes);
      itgt2spep = [intmax(cls)*ones(1,itgtmax,cls);...
                   intmin(cls)*ones(1,itgtmax,cls)]; % rows: sf, ef
      for iobj=1:nobj
        o = allobjs{iobj};
        itgtsI = o.pTrkiTgt;
        sfsI = o.startframes;
        efsI = o.endframes;
        tfhasres = sfsI<=efsI;
        itgt2spep(1,itgtsI(tfhasres)) = min(itgt2spep(1,itgtsI(tfhasres)),sfsI(tfhasres));
        itgt2spep(2,itgtsI(tfhasres)) = max(itgt2spep(2,itgtsI(tfhasres)),efsI(tfhasres));
      end
      
      sfsNew = itgt2spep(1,itgtsun);
      efsNew = itgt2spep(2,itgtsun);
      
      % 2. initialize new TrkFile with empty trkflds of right size
      % (nan-filled)
      objMerged = TrkFile(obj.npts,itgtsun,sfsNew,efsNew,obj.trkflds);
     
      % 3. init logical "isset" flag [nrms] indiciating whether frames set
      nfsNew = efsNew-sfsNew+1;      
      frmsAreSet = arrayfun(@(x)false(x,1),nfsNew,'uni',0);
      % frmsAreSet{jall} is nframes_jall x 1 logical indicator vec covering
      % objMerged.startframes(jall):objMerged.endframes(jall)
      
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
          itgt = o.pTrkiTgt(j);
          sp = o.startframes(j);
          ep = o.endframes(j);
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
            if strcmp(f,'pTrk')
              objMerged.(f){jall}(:,:,idxall) = o.(f){j}; 
            else
              objMerged.(f){jall}(:,idxall) = o.(f){j};
            end
          end
        end
      end
          
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
      end
    end
    
    function t = tableform(obj,varargin)
      ftonly = myparse(varargin,...
        'ftonly',false ...
        );
      
      assert(~obj.isfull);
      frm = arrayfun(@(x,y)(x:y)',obj.startframes,obj.endframes,'uni',0);
      iTgt = cellfun(@(x,y)repmat(x,numel(y),1),num2cell(obj.pTrkiTgt),frm,'uni',0);
      s = struct();
      s.frm = cat(1,frm{:});
      s.iTgt = cat(1,iTgt{:});
      if ftonly
        t = struct2table(s);
        return;
      end
      
      flds = obj.trkflds();
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        v = obj.(f);
        nd = cellfun(@ndims,v);
        assert(all(nd==nd(1)));
        nd = nd(1);
        v = cellfun(@(x)permute(x,[nd 1:nd-1]),v,'uni',0); % put 'frame' dim first
        % convert to 2d arrays (in particular for pTrk)
        v = cellfun(@(x)reshape(x,size(x,1),[]),v,'uni',0); 
        s.(f) = cat(1,v{:});
      end      
      
      t = struct2table(s);
    end
    
    function v = isalive(obj,f,itgt)
      
      
      if obj.isfull,
        % pTrk: [npttrked x 2 x nfrm x ntgt], like labeledpos
        if nargin < 3 || isempty(itgt),
          v = permute(any(any(~isnan(obj.pTrk(:,:,f,:)),1),2),[3,4,1,2]);
        elseif isempty(f),
          v = permute(any(any(~isnan(obj.pTrk(:,:,:,itgt)),1),2),[3,4,1,2]);
        else
          v = permute(any(any(~isnan(obj.pTrk(:,:,f,itgt)),1),2),[3,4,1,2]);
        end
        return;
      end
      
      if nargin < 3 || isempty(itgt),
        % all targets, v will be numel(f) x ntgts
        v = f(:) >= obj.startframes & f(:) <= obj.endframes;
      elseif isempty(f),
        % all frames, v will be nframes x numel(itgt)
        v = false(obj.nframes,numel(itgt));
        for i = 1:numel(itgt),
          v(i,obj.startframes(itgt(i)):obj.endframes(itgt(i))) = true;
        end
      else
        % v will be numel(f) x numel(itgt)
        v = f(:) >= obj.startframes(itgt) & f(:) <= obj.endframes(itgt);
      end
      
    end
    
    function v = frm2tltnnz(obj)
      if obj.isfull,
        v = nnz(any(any(~isnan(obj.pTrk(:,:,f,:)),1),2));
      else
        v = sum(obj.endframes-obj.startframes+1);
      end
    end
    
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
    
    % TODO: consider API that excludes ~tfhaspred vals
    function [tfhaspred,xy,tfocc] = getPTrkFrame(obj,f)
      % get tracking for particular frame
      %
      % f: scalar (absolute) frame index
      %
      % tfhaspred: [ntgt] logical vec, whether pred is present
      % xy: [npt x 2 x ntgt]
      % tfocc: [npt x ntgt]
      
      tfhaspred = obj.isalive(f).';
      %tfhaspred = obj.frm2tlt(f,:).';
      itgtsLive = find(tfhaspred);
      npt = obj.npts;
      ntgt = numel(obj.pTrk);
      xy = nan(npt,2,ntgt);
      tfocc = false(npt,ntgt);
      
      pcell = obj.pTrk;
      pcelltag = obj.pTrkTag;
      offs = 1-obj.startframes;
      
      for j=itgtsLive(:)'
        ptgt = pcell{j};
        ptag = pcelltag{j};
        idx = f + offs(j);
        xy(:,:,j) = ptgt(:,:,idx);
        tfocc(:,j) = ptag(:,idx);
      end
    end
    
    function [tfhaspred,xy,tfocc] = getPTrkFT(obj,f,iTgt)
      % get tracking for particular frame, tgt
      %
      % f: scalar (absolute) frame index
      % iTgt: scalar tracklet idx
      %
      % tfhaspred: logical scalar, whether pred is present
      % xy: [npt x 2]
      % tfocc: [npt x 1]
      
      tfhaspred = obj.isalive(f,iTgt);
      %tfhaspred = obj.frm2tlt(f,iTgt);
      if tfhaspred
        ptgt = obj.pTrk{iTgt};
        ptag = obj.pTrkTag{iTgt};
        offs = 1 - obj.startframes(iTgt);       
        idx = f + offs;
        xy = ptgt(:,:,idx);
        tfocc = ptag(:,idx);
      else
        npt = obj.npts;
        xy = nan(npt,2);
        tfocc = false(npt,1);  
      end
    end
    
    function frms = isLabeledT(obj,iTlt)
      frms = obj.startframes(iTlt):obj.endframes(iTlt);
    end
    
    function [xy,occ] = getPTrkTgt(obj,iTlt)
      % get tracking for particular target
      %
      % iTlt: tracklet index
      %
      % xy: [npt x 2 x nfrm] where nfrm=size(obj.frm2tlt,1)
      % occ: [npt x nfrm]
      
      if iTlt > obj.ntlts
        warningNoTrace('Tracklet %d exceeds available data.');
        xy = nan(obj.npts,2,0);
        occ = nan(obj.npts,0);
        return;
      end
      
      ptrkI = obj.pTrk{iTlt};
      npts = size(ptrkI,1);
      nfrm = obj.nframes;
      %nfrm = size(obj.frm2tlt,1);
      xy = nan(npts,2,nfrm);
      occ = false(npts,nfrm);
      f0 = obj.startframes(iTlt);
      f1 = obj.endframes(iTlt);
      idx = f0:f1;
      xy(:,:,idx) = ptrkI;
      occ(:,idx) = obj.pTrkTag{iTlt};
    end
    
    function xyaux = getPAuxTgt(obj,iTlt,ptrkfld,varargin)
      % get aux tracking timeseries (eg confidences) for particular tgt
      %
      % iTlt: tracklet index
      % ptrkfld: field, eg 'pTrkConf'
      %
      % xyaux: [npt x nfrm] auxiliary tracking
      
      missingok = myparse(varargin,...
        'missingok',false ...
        );
      
      npts = size(obj.pTrk{iTlt},1);
      nfrm = obj.nframes;
      %nfrm = size(obj.frm2tlt,1);
      xyaux = nan(npts,nfrm);
      
      if isprop(obj,ptrkfld)      
        pauxI = obj.(ptrkfld){iTlt};
        f0 = obj.startframes(iTlt);
        f1 = obj.endframes(iTlt);
        xyaux(:,f0:f1) = pauxI;
      elseif missingok
        % none; xyaux all nans
      else
        error('Unknown field ''%s''.',ptrkfld);
      end
    end
  end
  methods (Static)
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
        npt = size(t0.pTrk,2)/2;
        ptrk = reshape(t0.pTrk,n,npt,2,ntbl); 
        ptrk = permute(ptrk,[1 2 4 3]);
        ptrk = reshape(ptrk,n,[]);
        t0.pTrk = ptrk;
      end
      
      t = t0;
    end
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
          if strcmp(f,'pTrk') % branching off ndims(v) doesn't work if ntlt==1            
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
  end
  
  methods (Static)
    
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
          nFramesTracked = max(m.endframes - m.startframes) + 1;
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
