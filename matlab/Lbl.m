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
  
        tObj.setAllParams(lObj.trackParams); % does not set skel, flipLMEdges
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
      iscfg = isfield(s,'cfg');
      if iscfg,
        isMA = s.cfg.MultiAnimal;
      end
      
      FLDS = {'cfg' 'projname' 'projectFile' 'projMacros' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      if iscfg,
        CFG_GLOBS = {'Num' 'MultiAnimal' 'HasTrx'};
      else
        FLDS = setdiff(FLDS,{'cfg'});
      end
      TRACKERDATA_FLDS = {'sPrmAll' 'trnNetMode' 'trnNetTypeString'};
      if iscfg,
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
      else
        GLOBS = {};
        FLDSRM = {};
      end

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
      
      iscfg = isfield(s,'cfg');
      if iscfg,
        cfg = s.cfg;
        if isfield(s,'cropProjHasCrops'),
          cfg.HasCrops = s.cropProjHasCrops;
        end
      end
      if isfield(s,'movieInfoAll'),
        mia = cellfun(@(x)struct('NumRows',x.info.nr,...
          'NumCols',x.info.nc),s.movieInfoAll);
        for ivw=1:size(mia,2)
          nr = [mia(:,ivw).NumRows];
          nc = [mia(:,ivw).NumCols];
          assert(all(nr==nr(1) & nc==nc(1)),'Inconsistent movie dimensions for view %d',ivw);
        end
      end
      
      j = struct();
      if isfield(s,'projname'),
        j.ProjName = s.projname;
      end
      if isfield(s,'projectFile'),
        j.ProjectFile = s.projectFile;
      end
      if iscfg,
        j.Config = cfg;
      end
      if isfield(s,'movieInfoAll'),
        j.MovieInfo = mia(1,:);
      end
      if isfield(s,'cropProjHasCrops'),
        if cfg.HasCrops,
          ci = s.movieFilesAllCropInfo;
          nmov = numel(ci);
          nvw = s.cfg.NumViews;
          cropRois = nan(nmov,4,nvw);
          for imov=1:nmov
            for iv=1:nvw
              cropRois(imov,:,iv) = ci{imov}(iv).roi;
            end
          end
        else
          cropRois = [];
        end
        j.MovieCropRois = cropRois;
      end
      if isfield(s,'trackerClass'),
        assert(strcmp(s.trackerClass{2},'DeepTracker'));
      end
      if isfield(s,'trackerData'),
        if isempty(s.trackerData{1})
          j.TrackerData = s.trackerData{2};
        else
          j.TrackerData = cell2mat(s.trackerData);
        end
      end
      % KB 20220517 - added parameters here, as this is what is used when
      % saving json to file finally, wanted to reuse this. 
      % This output was never being used afaik.
      jse = jsonencode(j,'ConvertInfAndNaN',false);
    end
  end
  
end