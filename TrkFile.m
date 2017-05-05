classdef TrkFile < handle
  
  properties (Constant,Hidden)
    unsetVal = '__UNSET__';
  end
  
  properties
    pTrk = TrkFile.unsetVal; % [npttrked x 2 x nfrm], like labeledpos
    pTrkFull = TrkFile.unsetVal; % [npttrked x 2 x nRep x nfrm], with replicates
    pTrkTS = TrkFile.unsetVal; % [npttrked x nfrm], liked labeledposTS
    pTrkTag = TrkFile.unsetVal; % [npttrked x nfrm] cell, like labeledposTag
    pTrkiPt = TrkFile.unsetVal; % [npttrked]. point indices labeling rows of .pTrk*. If 
                           % npttrked=labeled.nLabelPoints, then .pTrkiPt=1:npttrked.
    pTrkFrm = TrkFile.unsetVal; % [nfrm]. frames tracked
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
      
      npttrk = size(ptrk,1);
      d = size(ptrk,2);
      nfrm = size(ptrk,3);
      if d~=2
        error('TrkFile:TrkFile','Expected d==2.');
      end
      obj.pTrk = ptrk;     

      if isequal(obj.pTrkFull,TrkFile.unsetVal)
        obj.pTrkFull = zeros(npttrk,2,0,nfrm);
      end
      nRep = size(obj.pTrkFull,3);
      validateattributes(obj.pTrkFull,{'numeric'},{'size' [npttrk 2 nRep nfrm]},'','pTrkFull');

      if isequal(obj.pTrkTS,TrkFile.unsetVal)
        obj.pTrkTS = zeros(npttrk,nfrm);
      end
      validateattributes(obj.pTrkTS,{'numeric'},{'size' [npttrk nfrm]},'','pTrkTS');
      
      if isequal(obj.pTrkTag,TrkFile.unsetVal)
        obj.pTrkTag = cell(npttrk,nfrm);
      end
      validateattributes(obj.pTrkTag,{'cell'},{'size' [npttrk nfrm]},'','pTrkTag');
      
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
