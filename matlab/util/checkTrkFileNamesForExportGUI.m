function [tfok, trkfiles] = checkTrkFileNamesForExportGUI(trkfiles, varargin)
  % Check/confirm trkfile names for export. If any trkfiles exist, ask
  % whether overwriting is ok; alternatively trkfiles may be
  % modified/uniqueified using datetimestamps.
  %
  % trkfiles (input): cellstr of proposed trkfile names (full paths).
  % Can be an array.
  %
  % tfok: if true, trkfiles (output) is valid, and user has said it is
  % ok to write to those files even if it is an overwrite.
  % trkfiles (output): cellstr, same size as trkfiles. .trk filenames
  % that are okay to write/overwrite to. Will match input if possible.

  noUI = myparse(varargin, ...
                 'noUI', false) ;

  tfexist = cellfun(@(x)(logical(exist(x, 'file'))), trkfiles(:)) ;
  tfok = true ;
  if any(tfexist)
    iExist = find(tfexist, 1) ;
    queststr = sprintf('One or more .trk files already exist, eg: %s.', trkfiles{iExist}) ;
    if noUI
      response = 'Add datetime to filenames' ;
      warningNoTrace('Labeler:trkFileNamesForExport', ...
                     'One or more .trk files already exist. Adding datetime to trk filenames.') ;
    else
      response = questdlg(queststr, 'Files exist', 'Overwrite', 'Add datetime to filenames', ...
                     'Cancel', 'Add datetime to filenames') ;
    end
    if isempty(response)
      response = 'Cancel' ;
    end
    switch response
      case 'Overwrite'
        % none; use trkfiles as-is
      case 'Add datetime to filenames'
        nowstr = datestr(now, 'yyyymmddTHHMMSS') ;
        [trkP, trkF] = cellfun(@fileparts, trkfiles, 'uni', 0) ;
        trkfiles = cellfun(@(x,y)(fullfile(x, [y '_' nowstr '.trk'])), trkP, trkF, 'uni', 0) ;
      otherwise
        tfok = false ;
        trkfiles = [] ;
    end
  end
end  % function
