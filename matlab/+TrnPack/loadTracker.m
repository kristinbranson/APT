function [fileinfo,trackerinfo,scfg] = loadTracker(cfgjsonfile)
  % Load metadata about a trained tracker from its tracker-config json
  % file, locating the training package files and the last network
  % checkpoint for each stage and view.
  fileinfo = struct;
  fileinfo.cfgjsonfile = cfgjsonfile;
  trackerinfo = struct;
  [fileinfo.packdir,timestampstr,ext] = fileparts(cfgjsonfile);
  assert(strcmp(ext,'.json'),'Must input json file with name <timestamp1>_<timestamp2>.json');
  scfg = TrnPack.hlpLoadJson(cfgjsonfile);
  trackerinfo.trnNetMode = {scfg.TrackerData.trnNetMode};
  trackerinfo.trnNetTypeString = {scfg.TrackerData.trnNetTypeString};
  nviews = scfg.Config.NumViews;
  trackerinfo.timestamps = regexp(timestampstr,'^(.*)_(.*)$','once','tokens');

  fileinfo.labelfile = fullfile(fileinfo.packdir,'loc.json');
  fileinfo.imdir = fullfile(fileinfo.packdir,TrnPack.SUBDIRIM);
  fileinfo.extrafiles = {};
  extraexts = {'.loc','.err','.cmd','aptsnapshot'};
  extrafiles = dir(fullfile(fileinfo.packdir,sprintf('%s*%s*%s',trackerinfo.timestamps{:})));
  extranames = {extrafiles.name};
  for i = 1:numel(extranames),
    [~,~,ext] = fileparts(extranames{i});
    if ~ismember(ext,extraexts),
      continue;
    end
    fileinfo.extrafiles{end+1} = fullfile(fileinfo.packdir,extranames{i});
  end

  % get the last checkpoint
  fileinfo.netfiles = cell(numel(trackerinfo.trnNetTypeString),nviews);
  fileinfo.trndirs = cell(numel(trackerinfo.trnNetTypeString),nviews);
  for i = 1:numel(trackerinfo.trnNetTypeString),
    netdir = fullfile(fileinfo.packdir,trackerinfo.trnNetTypeString{i});
    if ~exist(netdir,'dir'),
      continue;
    end
    for view = 1:nviews,
      viewdir = fullfile(netdir,sprintf('view_%d',view-1));
      if ~exist(viewdir,'dir'),
        continue;
      end
      trndir = fullfile(viewdir,trackerinfo.timestamps{1});
      if ~exist(trndir,'dir'),
        continue;
      end
      fileinfo.trndirs{i,view} = trndir;
      lastcheckpointfile = fullfile(trndir,'last_checkpoint');
      if ~exist(lastcheckpointfile,'file'),
        continue;
      end
      fid = fopen(lastcheckpointfile,'r');
      netfile = '';
      while true,
        s = fgetl(fid);
        if ~ischar(s),
          break;
        end
        s = strtrim(s);
        if ~isempty(s),
          netfile = s;
          break;
        end
      end
      fclose(fid);
      if isempty(netfile),
        continue;
      end
      fileinfo.netfiles{i,view} = netfile;
    end
  end
end % function
