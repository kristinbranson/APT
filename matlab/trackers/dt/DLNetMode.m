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
    singleAnimal ('sa', false, false, 0, false, false)
    multiAnimalBU ('bu', true, false, 0, false, true)
    multiAnimalTDDetectObj ('tddobj', true, true, 1, false, true)
    multiAnimalTDDetectHT ('tddht', true, true, 1, true, true) 
    multiAnimalTDPoseTrx ('tdptrx', true, true, 2, false, false)
    multiAnimalTDPoseObj ('tdpobj', true, true, 2, false, true)
    multiAnimalTDPoseHT ('tdpht', true, true, 2, true, true)
  end
  properties
    shortCode % for eg logfiles
    isMA
    isTopDown
    topDownStage % 1 or 2
    isHeadTail 
    isTrnPack
  end
  methods 
    function obj = DLNetMode(code,ma,topdown,stage,ht,istp)
      obj.shortCode = code;
      obj.isMA = ma;
      obj.isTopDown = topdown;
      obj.topDownStage = stage;
      obj.isHeadTail = ht;
      obj.isTrnPack = istp;
    end
  end
end