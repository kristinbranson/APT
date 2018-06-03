classdef TrkFile < handle
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
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
        obj.(prop) = val;
      end
      
      [npttrk,d,nfrm,ntgt] = size(ptrk);
      if d~=2
        error('TrkFile:TrkFile','Expected d==2.');
      end
      obj.pTrk = ptrk;

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
    
    function trkfile = mergePartial(obj,obj2)
      % Merge trkfile into current trkfile. Doesn't merge .pTrkFull* fields 
      % (for now).
      %
      % obj2 TAKES PRECEDENCE when tracked frames/data overlap
      %
      % obj/obj2: trkfile objs. obj.pTrk and obj2.pTrk must have the same
      % size.
      
      assert(isscalar(obj) && isscalar(obj2));
      %assert(isequal(size(obj.pTrk),size(obj2.pTrk)),'Size mismatch.');
      assert(isequal(obj.pTrkiPt,obj2.pTrkiPt),'.pTrkiPt mismatch.');
      assert(isequal(obj.pTrkiTgt,obj2.pTrkiTgt),'.pTrkiTgt mismatch.');
      
      if ~isempty(obj.pTrkFull) || ~isempty(obj2.pTrkFull)
        warningNoTrace('.pTrkFull contents discarded.');
      end
      
      frmComon = intersect(obj.pTrkFrm,obj2.pTrkFrm);
      frmUnion = union(obj.pTrkFrm,obj2.pTrkFrm);
      nComon = numel(frmComon);
      if nComon>0
        warningNoTrace('TrkFiles share common tracked frames. Overwriting.');
      end
      
      [~,loc] = ismember(obj.pTrkFrm,frmUnion);
      [~,loc2] = ismember(obj2.pTrkFrm,frmUnion);     
      
      npttrk = numel(obj.pTrkiPt);
      nfrm = numel(frmUnion);
      ntgt = numel(obj.pTrkiTgt);
      
      pTrk = nan(npttrk,2,nfrm,ntgt);
      pTrk(:,:,loc,:) = obj.pTrk;
      pTrk(:,:,loc2,:) = obj2.pTrk;
      obj.pTrk = pTrk;
      pTrkTS = nan(npttrk,nfrm,ntgt);
      pTrkTS(:,loc,:) = obj.pTrkTS;
      pTrkTS(:,loc2,:) = obj2.pTrkTS;
      obj.pTrkTS = pTrkTS;
      pTrkTag = nan(npttrk,nfrm,ntgt);
      pTrkTag(:,loc,:) = obj.pTrkTag;
      pTrkTag(:,loc2,:) = obj2.pTrkTag;
      obj.pTrkTag = pTrkTag;
      
      %obj.pTrkiPt = obj.pTrkiPt; unchanged
      obj.pTrkFrm = frmUnion;
      %obj.pTrkiTgt = obj.pTrkiTgt; unchanged
      %obj.pTrkFull unchanged
      %obj.pTrjFullFT unchanged
      
      if iscell(obj.trkInfo)
        obj.trkInfo{end+1} = obj2.trkInfo;
      else
        obj.trkInfo = {obj.trkInfo obj2.trkInfo};
      end
    end
    
  end
  
  methods (Static)

    function trkfileObj = load(filename)
      s = load(filename,'-mat');
      s = TrkFile.modernizeStruct(s);
      if ~isfield(s,'pTrk')
        error('TrkFile:load',...
          'File ''%s'' is not a valid saved trkfile structure.',filename);
      end
      pTrk = s.pTrk;
      pvs = struct2pvs(rmfield(s,'pTrk'));
      trkfileObj = TrkFile(pTrk,pvs{:});
    end
    
    function trkfileObj = loadsilent(filename)
      % ignore fields of struct that aren't TrkFile props. For 3rd-party
      % generated Trkfiles
      s = load(filename,'-mat');
      s = TrkFile.modernizeStruct(s);
      pTrk = s.pTrk;      
      
      mc = meta.class.fromName('TrkFile');
      propnames = {mc.PropertyList.Name}';
      fns = fieldnames(s);
      tfunrecog = ~ismember(fns,propnames);
      s = rmfield(s,fns(tfunrecog)); 
      pvs = struct2pvs(rmfield(s,'pTrk'));
      
      trkfileObj = TrkFile(pTrk,pvs{:});
    end

    function s = modernizeStruct(s)
      % s: struct loaded from trkfile saved to matfile
      if iscell(s.pTrkTag)
        s.pTrkTag = strcmp(s.pTrkTag,'occ');
      end
    end
    
  end
  
end
