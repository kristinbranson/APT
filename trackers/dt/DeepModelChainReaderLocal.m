classdef DeepModelChainReaderLocal < DeepModelChainReader
  methods
    function maxiter = getMostRecentModel(obj,dmc)
      maxiter = nan;
      
      modelglob = dmc.trainModelGlob;
      modelfiles = mydir(fullfile(dmc.dirModelChainLnx,modelglob));
      if isempty(modelfiles),
        return;
      end
      
      maxiter = -1;
      for i = 1:numel(modelfiles),
        iter = DeepModelChainOnDisk.getModelFileIter(modelfiles{i});
        if iter > maxiter,
          maxiter = iter;
        end
      end
    end
  end
end