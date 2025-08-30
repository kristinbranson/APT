function trackWriteListFile(obj, movFileNativePath, movidx, tMFTConc, listFileNativePath, varargin)
  % Write the .json file that specifies the list of frames to track.
  % File paths in movFileNativePath and listFileNativePath should be absolute native paths.
  % Throws if unable to write complete file.

  % Get optional args
  [trxFilesLcl, croprois] = ...
    myparse(varargin,...
            'trxFiles',cell(0,1),...
            'croprois',cell(0,1) ...
            );

  % Check arg types
  assert(iscell(movFileNativePath)) ;
  assert(isempty(movFileNativePath) || isstringy(movFileNativePath{1})) ;
  assert(iscolumn(movidx)) ;
  assert(isnumeric(movidx)) ;
  assert(istable(tMFTConc)) ;
  assert(isstringy(listFileNativePath)) ;
  assert(iscell(trxFilesLcl)) ;
  assert(isempty(trxFilesLcl) || isstringy(trxFilesLcl{1})) ;
  assert(iscell(croprois)) ;
  assert(isempty(croprois) || (isnumeric(croprois{1}) && isequal(size(croprois{1}),[1 4])) ) ;

  % Check dimensions
  [nmoviesets, nviews] = size(movFileNativePath) ;
  assert(size(movidx,1) == nmoviesets) ;
  
  % Create listinfo struct, populate some fields
  ismultiview = (nviews > 1) ;      
  listinfo = struct() ;
  if ismultiview,
    assert(isempty(croprois) || (size(croprois,1) == nmoviesets)) ;
    listinfo.movieFiles = cell(nmoviesets,1);
    for i = 1:nmoviesets,
      listinfo.movieFiles{i} = wsl_path_from_native(movFileNativePath(i,:)) ;
    end
    listinfo.trxFiles = cell(size(trxFilesLcl,1),1);
    for i = 1:size(trxFilesLcl,1),
      listinfo.trxFiles{i} = wsl_path_from_native(trxFilesLcl(i,:)) ;
    end
    listinfo.cropLocs = cell(size(croprois,1),1);
    for i = 1:size(croprois,1),
      listinfo.cropLocs{i} = croprois(i,:);
    end
  else
    listinfo.movieFiles = wsl_path_from_native(movFileNativePath) ;
    listinfo.trxFiles = wsl_path_from_native(trxFilesLcl) ;
    listinfo.cropLocs = croprois ;
  end

  % which movie index does each row correspond to?
  % assume first movie is unique
  [ism,idxm] = ismember(tMFTConc.mov(:,1),movidx(:,1));
  assert(all(ism));
 
  % Populate listinfo.toTrack
  listinfo.toTrack = cell(0,1);
  for mi = 1:nmoviesets,
    idx1 = find(idxm==mi);
    if isempty(idx1),
      continue
    end
    [t,~,idxt] = unique(tMFTConc.iTgt(idx1));
    for ti = 1:numel(t),
      idx2 = idxt==ti;
      idxcurr = idx1(idx2);
      f = unique(tMFTConc.frm(idxcurr));
      df = diff(f);
      istart = [1;find(df~=1)+1];
      iend = [istart(2:end)-1;numel(f)];
      for i = 1:numel(istart),
        if istart(i) == iend(i),
          fcurr = f(istart(i));
        else
          fcurr = [f(istart(i)),f(iend(i))+1];
        end
        listinfo.toTrack{end+1,1} = {mi,t(ti),fcurr};
      end
    end
  end

  % Encode listinfo struct to json and write to file
  listinfo_as_json_string = jsonencode(listinfo) ;
  obj.writeStringToFile(listFileNativePath, listinfo_as_json_string) ;  % throws if unable to write file
end  % function