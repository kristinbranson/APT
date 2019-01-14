classdef BgTrackWorkerObjAWS < BgTrackWorkerObj  
  properties
    awsEc2 % Instance of AWSec2
  end
  methods    
    function obj = BgTrackWorkerObjAWS(aws,varargin)
      obj@BgTrackWorkerObj(varargin{:});
      obj.awsEc2 = aws;
    end    
    
    % same as in BgTrainWorkerObjAWS
    function tf = fileExists(obj,f)
      tf = obj.awsEc2.remoteFileExists(f);
    end
    function tf = errFileExistsNonZeroSize(obj,errFile)
      tf = obj.awsEc2.remoteFileExists(errFile,'reqnonempty',true);
    end    
    function s = fileContents(obj,f)
      s = obj.awsEc2.remoteFileContents(f);
    end
  end
end