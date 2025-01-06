classdef DeepModelChainReaderAWS < DeepModelChainReader
  % Vestigial class to support old .lbl files  
  properties 
    awsec2
  end
  
  methods (Access=protected)
    function obj2 = copyElement(obj)
      obj2 = copyElement@DeepModelChainReader(obj);
      if ~isempty(obj.awsec2)
        obj2.awsec2 = copy(obj.awsec2);
      end
    end
  end
  
  methods   
    function obj = DeepModelChainReaderAWS(aws)
      assert(isa(aws,'AWSec2'));
      assert(aws.isInstanceIDSet);
      obj.awsec2 = aws; % awsec2 is specified and so .instanceID is immutable
    end    
  end
end
