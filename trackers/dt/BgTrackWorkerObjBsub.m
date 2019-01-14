classdef BgTrackWorkerObjBsub < BgTrackWorkerObj  
  methods    
    function obj = BgTrackWorkerObjBsub(varargin)
      obj@BgTrackWorkerObj(varargin{:});
    end    
    function tf = fileExists(~,file)
      tf = exist(file,'file')>0;
    end        
    function tf = errFileExistsNonZeroSize(~,errFile)
      tf = BgTrainWorkerObjLocalFilesys.errFileExistsNonZeroSizeStc(errFile);
    end  
    function s = fileContents(~,file)
      lines = readtxtfile(file);
      s = sprintf('%s\n',lines{:});
    end
  end
end