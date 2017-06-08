classdef TrkFile < handle
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
  end
  
  properties
    pTrk = TrkFile.unsetVal; % [npttrked x 2 x nfrm x ntgt], like labeledpos
    pTrkFull = TrkFile.unsetVal; % [npttrked x 2 x nRep x nfrm x ntgt], with replicates
    pTrkTS = TrkFile.unsetVal; % [npttrked x nfrm x ntgt], liked labeledposTS
    pTrkTag = TrkFile.unsetVal; % [npttrked x nfrm x ntgt] cell, like labeledposTag
    pTrkiPt = TrkFile.unsetVal; % [npttrked]. point indices labeling rows of .pTrk*. If 
                           % npttrked=labeled.nLabelPoints, then .pTrkiPt=1:npttrked.
    pTrkFrm = TrkFile.unsetVal; % [nfrm]. frames tracked
    pTrkiTgt = TrkFile.unsetVal; % [ntgt]. targets (1-based Indices) tracked
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

      if isequal(obj.pTrkFull,TrkFile.unsetVal)
        obj.pTrkFull = zeros(npttrk,2,0,nfrm,ntgt);
      end
      nRep = size(obj.pTrkFull,3);
      validateattributes(obj.pTrkFull,{'numeric'},{'size' [npttrk 2 nRep nfrm ntgt]},'','pTrkFull');

      if isequal(obj.pTrkTS,TrkFile.unsetVal)
        obj.pTrkTS = zeros(npttrk,nfrm,ntgt);
      end
      validateattributes(obj.pTrkTS,{'numeric'},{'size' [npttrk nfrm ntgt]},'','pTrkTS');
      
      if isequal(obj.pTrkTag,TrkFile.unsetVal)
        obj.pTrkTag = cell(npttrk,nfrm,ntgt);
      end
      validateattributes(obj.pTrkTag,{'cell'},{'size' [npttrk nfrm ntgt]},'','pTrkTag');
      
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

    end
    
    function save(obj,filename)
      % Saves to filename; ALWAYS OVERWRITES!!
      
      warnst = warning('off','MATLAB:structOnObject');
      s = struct(obj);
      warning(warnst);
      s = rmfield(s,'unsetVal'); %#ok<NASGU>      
      save(filename,'-mat','-struct','s');      
    end
    
  end
  
end
