classdef TrainData < handle
  properties    
    Name    % Name of this TrainData
    
    I       % [N] column cell vec, images
    pGT     % [NxD] GT shapes for I
    bboxes  % [Nx2d] bboxes for I 
    
    iTrn    % [Ntrn] indices into I for training set
    iTst    % [Ntst] indices into I for test set    
  end
  properties (Dependent)
    N
    d
    D
    nfids
    
    ITrn
    ITst
    pGTTrn
    pGTTst
    bboxesTrn
    bboxesTst
  end
  methods 
    function v = get.N(obj)
      v = numel(obj.I);
    end
    function v = get.d(obj)
      v = size(obj.bboxes,2)/2;
    end
    function v = get.D(obj)
      v = size(obj.pGT,2);
    end
    function v = get.nfids(obj)
      v = obj.D/obj.d;
    end
    function v = get.ITrn(obj)
      v = obj.I(obj.iTrn,:);
    end
    function v = get.ITst(obj)
      v = obj.I(obj.iTst,:);
    end
    function v = get.pGTTrn(obj)
      v = obj.pGT(obj.iTrn,:);
    end
    function v = get.pGTTst(obj)
      v = obj.pGT(obj.iTst,:);      
    end
    function v = get.bboxesTrn(obj)
      v = obj.bboxes(obj.iTrn,:);
    end
    function v = get.bboxesTst(obj)
      v = obj.bboxes(obj.iTst,:);
    end
  end
  methods
    function obj = TrainData(Is,p,bb)
      assert(iscolumn(Is) && iscell(Is));
      N = numel(Is);
      assert(size(p,1)==N);
      assert(size(bb,1)==N);
      
      obj.I = Is;
      obj.pGT = p;
      obj.bboxes = bb;
    end
    function viz(obj,varargin)
      Shape.viz(obj.I,obj.pGT,struct('nfids',obj.D/obj.d,'D',obj.D),varargin{:});
    end
  end
end