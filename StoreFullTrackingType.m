classdef StoreFullTrackingType < uint32
  enumeration
    NONE(0) % don't store full tracking
    FINALITER(10) % store final replicate cloud 
    ALLITERS(20) % store full replicate cloud, all iterations
  end
end
