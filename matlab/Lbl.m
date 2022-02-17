classdef Lbl
  
  methods (Static) % stripped lbl
    function s = createStrippedLblsUseTopLevelTrackParams(lObj,iTrkers,...
        varargin)
      % Create/save a series of stripped lbls based on current Labeler proj
      %
      % lObj: Labeler obj with proj loaded
      % iTrkers: vector of tracker indices for which stripped lbl will be 
      %   saved
      %
      % s: cell array of stripped lbls
      %
      % This method exists bc:
      % - Strictly speaking, stripped lbls are net-specific, as setting
      % base tracking parameters onto a DeepTracker obj has hooks/codepath
      % for mutating params.
      % - Sometimes, you want to generate a stripped lbl from the top-level
      % params which are not yet set on a particular tracker.
      %
      % This method is here rather than Labeler bc Labeler is getting big.
            
      [docompress,dosave] = myparse(varargin,...
        'docompress',true, ...
        'dosave',true ... save stripped lbls (loc printed)
        );
      
      ndt = numel(iTrkers);
      s = cell(ndt,1);
      for idt=1:ndt
        itrker = iTrkers(idt);
        lObj.trackSetCurrentTracker(itrker);
        tObj = lObj.tracker;
  
        tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges
        sthis = tObj.trnCreateStrippedLbl();
        if docompress
          sthis = Lbl.compressStrippedLbl(sthis);
        end
        
        s{idt} = sthis;
        
        if dosave
          fsinfo = lObj.projFSInfo;
          [lblP,lblS] = myfileparts(fsinfo.filename);
          sfname = sprintf('%s_%s.lbl',lblS,tObj.algorithmName);
          sfname = fullfile(lblP,sfname);
          save(sfname,'-mat','-struct','sthis');
          fprintf(1,'Saved %s\n',sfname);
        end
      end
      
    end
    function s = compressStrippedLbl(s,varargin)
      isMA = s.cfg.MultiAnimal;
      
      CFG_GLOBS = {'Num' 'MultiAnimal' 'HasTrx'};
      FLDS = {'cfg' 'projname' 'projectFile' 'projMacros' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      TRACKERDATA_FLDS = {'sPrmAll' 'trnNetMode' 'trnNetTypeString'};
      if isMA
        GLOBS = {'movieFilesAll' 'movieInfoAll' 'trxFilesAll'};
        FLDSRM = {'projMacros'};
      else
        GLOBS = {'labeledpos' 'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'preProcData'};
        FLDSRM = { ... % 'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
                  'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT'};
      end
      
      fldscfg = fieldnames(s.cfg);      
      fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
      s.cfg = structrestrictflds(s.cfg,fldscfgkeep);

      for i=1:numel(s.trackerData)
        if ~isempty(s.trackerData{i})
          s.trackerData{i} = structrestrictflds(s.trackerData{i},...
                                                TRACKERDATA_FLDS);
        end
      end
      
      flds = fieldnames(s);
      fldskeep = flds(startsWith(flds,GLOBS));
      fldskeep = [fldskeep(:); FLDS(:)];
      fldskeep = setdiff(fldskeep, FLDSRM);
      s = structrestrictflds(s,fldskeep);
    end
    function [jse,j] = jsonifyStrippedLbl(s)
      % s: compressed stripped lbl (output of compressStrippedLbl)
      %
      % jse: jsonencoded struct
      % j: raw struct
      
      cfg = s.cfg;
      cfg.HasCrops = s.cropProjHasCrops;
      mia = cellfun(@(x)struct('NumRows',x.info.nr,...
                               'NumCols',x.info.nc),s.movieInfoAll);
      for ivw=1:size(mia,2)
        nr = [mia(:,ivw).NumRows];
        nc = [mia(:,ivw).NumCols];
        assert(all(nr==nr(1) & nc==nc(1)),'Inconsistent movie dimensions for view %d',ivw);        
      end
      
      j = struct();
      j.ProjName = s.projname;
      j.ProjectFile = s.projectFile;
      j.Config = cfg;
      j.MovieInfo = mia(1,:);
      if cfg.HasCrops
        ci = s.movieFilesAllCropInfo;
        nmov = numel(ci);
        cropRois = nan(nmov,4);
        for imov=1:nmov
          cropRois(imov,:) = ci{imov}.roi;
        end
      else
        cropRois = [];        
      end
      j.MovieCropRois = cropRois;
      assert(strcmp(s.trackerClass{2},'DeepTracker'));
      if isempty(s.trackerData{1})
        j.TrackerData = s.trackerData{2};
      else
        j.TrackerData = cell2mat(s.trackerData);
      end
      
      jse = jsonencode(j);
    end
  end
  
end