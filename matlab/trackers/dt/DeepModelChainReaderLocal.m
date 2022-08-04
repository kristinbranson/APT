classdef DeepModelChainReaderLocal < DeepModelChainReader
  methods
    function  tf = getModelIsRemote(obj)
      tf = false;
    end
    function [maxiter,idx] = getMostRecentModel(obj,dmc,varargin)
      [modelglob,idx] = dmc.trainModelGlob(varargin{:});
      [dirModelChainLnx] = dmc.dirModelChainLnx(idx);

      maxiter = nan(1,numel(idx));
      for i = 1:numel(idx),
        modelfiles= mydir(fullfile(dirModelChainLnx{i},modelglob{i}));
        if isempty(modelfiles),
          continue;
        end
        for j = 1:numel(modelfiles),
          iter = DeepModelChainOnDisk.getModelFileIter(modelfiles{j});
          if ~isempty(iter),
            maxiter(i) = max(maxiter(i),iter);
          end
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