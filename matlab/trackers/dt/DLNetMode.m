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
    singleAnimal (false, 'sa')
    multiAnimalBU (true, 'bu')
    multiAnimalTDDetectObj (true, 'tddobj')
    multiAnimalTDDetectHT (true, 'tddht') 
    multiAnimalTDPoseTrx (true, 'tdptrx')
    multiAnimalTDPoseObj (true, 'tdpobj')
    multiAnimalTDPoseHT (true, 'tdpht')
  end
  properties
    isMA
    shortCode % for eg logfiles
  end
  methods 
    function obj = DLNetMode(tf,code)
      obj.isMA = tf;
      obj.shortCode = code;
    end
  end
end