classdef DLNetMode < handle   
  % While DLNetType captures a network type, the DLNetType by itself does
  % not tell the entire story as a given network may be used in various 
  % ways for pose tracking.
  %
  % For instance, a given deepnet could be used for single-animal tracking,
  % multi-animal tracking with trx (as stage2), multi-animal tracking to 
  % detect HT pairs (as stage1), and so on.
  %
  % A given training job requires
  % - DLBackEndClass
  % - DLNetType
  % - DLNetMode
  
  enumeration 
    singleAnimal ('sa', false, false, 0, false, false, true)
    multiAnimalBU ('bu', true, false, 0, false, false, true)
    multiAnimalTDDetectObj ('tddobj', true, true, 1, false, true, true)
    multiAnimalTDDetectHT ('tddht', true, true, 1, true, false, true) 
    multiAnimalTDPoseTrx ('tdptrx', true, true, 2, false, false, true)
    multiAnimalTDPoseObj ('tdpobj', true, true, 2, false, true, true)
    multiAnimalTDPoseHT ('tdpht', true, true, 2, true, false, true)
  end
  properties
    shortCode % for eg logfiles
    isMA
    isTopDown
    topDownStage % 1 or 2
    isHeadTail 
    isObjDet
    isTrnPack
  end
  properties (Dependent)
    is_multi        % config param for backend
    multi_crop_ims  % "
    multi_only_ht   % "
    isTwoStage
  end
  methods
    function v = get.is_multi(obj)
      v = obj.isMA && ~(obj.isTopDown && obj.topDownStage==2);
    end
    function v = get.multi_crop_ims(obj)
      v = obj.isMA && ~(obj.isTopDown && obj.isObjDet);
    end
    function v = get.multi_only_ht(obj)
      v = obj.isMA && obj.isTopDown && obj.isHeadTail && obj.topDownStage==1;
    end
    function v = get.isTwoStage(obj)
      v = obj.isTopDown && (obj.isHeadTail || obj.isObjDet);
    end
  end
  methods 
    function obj = DLNetMode(code,ma,topdown,stage,ht,od,istp)
      obj.shortCode = code;
      obj.isMA = ma;
      obj.isTopDown = topdown;
      obj.topDownStage = stage;
      obj.isHeadTail = ht;
      obj.isObjDet = od;
      obj.isTrnPack = istp;
    end
    function tf = netTypeMatches(obj,nettypes,nstages,stage)
      % tf = netTypeMatches(obj,nettypes,nstages,stage)
      % Tests whether the input nettypes are instances of this netmode
      % nettypes: array of DLNetTypes
      % nstages: total number of stages
      % stage: which stage this is 
      
      isma = [nettypes.isMultiAnimal] ;
      is_bbox = false(1,numel(nettypes)) ;
      for dndx = 1:numel(nettypes)          
        is_bbox(dndx) = nettypes(dndx).isMultiAnimal && startsWith(char(nettypes(dndx)),'detect_') ;
      end  % for
      tf = (isma == obj.isMA) & (obj.isObjDet == is_bbox) & (obj.isHeadTail == ~is_bbox) & ...
        ((nstages > 1) == obj.isTopDown) & (stage == obj.topDownStage);

    end

  end
end
