function importTrkResave(lc,lblname)
% Import tracking results into an APT project and resave.
%
% Use when you have re-run tracking but want to keep project structure
% (backup project files first).
%
% Run like this:
%   lc = LabelerController();
%   importTrkResave(lc,'/full/path/to/lbl/file')
%
% You will be prompted to select a .trk file for each movie (and view).

% set to 1 if want to put date at end of filename and keep old file
% set to 0 if just want to overwrite old file
putDateAtEndOfFilename = 0 ;

lc.load(lblname) ;
fprintf(1, 'Loaded proj: %s\n', lblname) ;
labeler = lc.labeler_ ;
for iMov = 1:labeler.nmovies
  lc.labelImportTrkPromptGenericSimple(iMov, 'importTrackingResults') ;
end

if putDateAtEndOfFilename == 1
  time = clock() ;  %#ok<CLOCK>
  [p, f, e] = fileparts(lblname) ;
  lblname2 = fullfile(p, [f, '_trk_', num2str(time(1)), num2str(time(2)), ...
                          num2str(time(3)), e]) ;
else
  lblname2 = lblname ;
end

lc.save(lblname2) ;
fprintf(1, 'Saved proj: %s\n', lblname2) ;

