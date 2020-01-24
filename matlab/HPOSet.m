classdef HPOSet

  properties
    rootDir % root directory of folds/hpo data
    foldNames % [nfold] cellstr of fold names
    foldJsons % [nfold] cellstr of fold jsons
    hpoObjs % [nfold] vector of HPOptim objects    
  end
  
  methods
    
    function obj = HPOSet(varargin)
      [rootdir,nview] = myparse(varargin,...
        'rootdir',pwd,...
        'nview',1....
      );
    
      
      % look for first-level subdirs with jsons. these are our folds
      foldsubdirs = cell(0,1);
      foldjsons = cell(0,1); 
      hpoobjs = cell(0,1);
      dd = dir(rootdir);
      dd = dd([dd.isdir]);
      for i=1:numel(dd)
        subdirS = dd(i).name;
        subdir = fullfile(rootdir,subdirS);
        jsonpat = fullfile(subdir,'*.json');        
        ddjson = dir(jsonpat);
        switch numel(ddjson)
          case 0
            % none
          case 1
            jsonS = ddjson.name;
            fprintf('Found fold subdir %s with json %s.\n',subdirS,jsonS);
            foldsubdirs{end+1,1} = subdirS; %#ok<AGROW>
            foldjsons{end+1,1} = jsonS; %#ok<AGROW>    
            try
              hpoobjs{end+1,1} = HPOptim(nview,'baseDir',fullfile(rootdir,subdirS)); %#ok<AGROW>
            catch ME
              fprintf(2,'Error caught in subdir %s: %s\n\n.Skipping subdir\n',...
                subdirS,ME.message);
              hpoobjs{end+1,1} = HPOptim; %#ok<AGROW>
              continue;
            end              
          otherwise
            warningNoTrace('Multiple jsons found in subdir %s. Skipping...',subdirS);
        end
      end
      
      obj.rootDir = rootdir;
      obj.foldNames = foldsubdirs;
      obj.foldJsons = foldjsons;
      obj.hpoObjs = cat(1,hpoobjs{:});     
    end
    
    function diffs = showPrmDiffs(obj)
      hpos = obj.hpoObjs;
      nfold = numel(hpos);      
      
      for i=1:nfold
        h = hpos(i);
        fprintf(1,'Fold %s has %d rounds.\n',obj.foldNames{i},h.nround);
      end
      
      prms0 = arrayfun(@(x)x.prms{1},hpos,'uni',0);
      prms1 = arrayfun(@(x)x.prms{end},hpos,'uni',0);
      assert(isequaln(prms0{:}));
      diffs = cellfun(@(x,y)structdiff(x,y,'quiet',true),prms0,prms1,'uni',0);
      
      %assert(nfold==3,'3 rounds only for the moment.');
      
      alldiffs = cat(1,diffs{:});
      alldiffsUn = unique(alldiffs);
      alldiffsUnCnt = cellfun(@(x)nnz(strcmp(x,alldiffs)),alldiffsUn);
      maxCnt = max(alldiffsUnCnt);      
      for cnt=maxCnt:-1:1
        fprintf('\nDiffs found in %d folds:\n',cnt);
        alldiffsUnIdx = find(alldiffsUnCnt==cnt);
        for idx=alldiffsUnIdx(:)'
          diff = alldiffsUn{idx};
          fprintf(1,'  %s\n',diff);
          p0 = prms0{1};  
          val0 = eval(['p0' diff]);
          valstr = num2str(val0);
          fprintf(1,'    Orig0: prm0: %s\n',valstr);
          for ifold=1:nfold
            if any(strcmp(diff,diffs{ifold}))
              p1 = prms1{ifold};
              val1 = eval(['p1' diff]);
              valstr = num2str(val1);
              fprintf(1, '    Fold%d: %s\n',ifold,valstr);
            end
          end
        end
      end      
    end
    
    function [hFigs,scoreSRs] = plotTrnTrkErrWithSelect(obj,varargin)
      h = obj.hpoObjs;
      
      fignum0 = 11;
      hFigs = [];
      scoreSRs = cell(numel(h),1);      
      for i=1:numel(h)
        [hFig,scoreSR] = h(i).plotTrnTrkErrWithSelect('fignums',fignum0+[1 2]);
        fignum0 = fignum0+2;
        hFigs = cat(2,hFigs,hFig(:));
        scoreSRs{i} = scoreSR';
      end
    end
    
  end
  
end