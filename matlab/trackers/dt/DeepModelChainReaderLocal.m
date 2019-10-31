classdef DeepModelChainReaderLocal < DeepModelChainReader
  methods
    function  tf = getModelIsRemote(obj)
      tf = false;
    end
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
    function lsProjDir(obj,dmc)
      ls('-al',dmc.dirProjLnx);
    end
    function lsModelChainDir(obj,dmc)
      ls('-al',dmc.dirModelChainLnx);
    end
    function lsTrkDir(obj,dmc)
      ls('-al',dmc.dirTrkOutLnx);
    end
  end
end