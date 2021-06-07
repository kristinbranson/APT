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
    singleAnimal (false)
    multiAnimalBU (true)
    multiAnimalTDDetectObj (true)
    multiAnimalTDDetectHT (true)    
    multiAnimalTDPoseTrx (true)
    multiAnimalTDPoseObj (true)
    multiAnimalTDPoseHT (true)
  end
  properties
    isMA
  end
  methods 
    function obj = DLNetMode(tf)
      obj.isMA = tf;
    end
  end
end