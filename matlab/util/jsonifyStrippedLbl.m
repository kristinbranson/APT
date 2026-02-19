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
end % function
