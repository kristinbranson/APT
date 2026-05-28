function test_mmc_change_path()
% Test the "Change Path..." button in MovieManagerController.
% Exercises:
%   1. pbChangePath enablement as a function of which cell is selected.
%   2. MovieManagerController.determineSelectedPathCell_() correctness.
%   3. Labeler.collectPrefixReplacementCandidates() with all literal
%      paths.
%   4. The same with some macroized paths --- those should be excluded
%      from the candidate list.
%   5. End-to-end: relocate one cell, compute candidates, apply them,
%      and verify both the model state and the table styling.
%
% Modeled on test_mmc_file_existence_styles, which loads the same
% project.

% Define the project path
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3_lightly_trained.lbl' ;
[project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

% Launch APT and create cleaners
[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ; 
cleaner2 = onCleanup(@()(delete(labeler))) ;

% Launch the Movie Manager
controller.menu_file_managemovies_actuated_([], []) ;
mmc = controller.movieManagerController_ ;
if isempty(mmc) || ~isvalid(mmc.hFig)
  error('test_mmc_change_path:noMMC', 'Movie Manager did not open') ;
end

% Sanity check: this project should be single-view, with trx, with no macros.
assert(labeler.nview == 1, 'Expected single-view project') ;
assert(labeler.hasTrx, 'Expected project with trx files') ;
assert(isempty(fieldnames(labeler.projMacros)), 'Expected no project macros') ;

% Snapshot original state so each section starts from a clean slate.
movieFilesOriginal = labeler.movieFilesAll ;
movieFilesGTOriginal = labeler.movieFilesAllGT ;
trxFilesOriginal = labeler.trxFilesAll ;
trxFilesGTOriginal = labeler.trxFilesAllGT ;


% --- Section 1: Test pbChangePath enablement is several scenarios ---------------
% 1a. No selection -> disabled.
simulateMainTableSelection(mmc, zeros(0, 2)) ;
assertEnable(mmc.pbChangePath, false, 'no selection') ;

% 1b. Movie cell -> enabled.
simulateMainTableSelection(mmc, [1, 1]) ;
assertEnable(mmc.pbChangePath, true, 'movie cell selected') ;

% 1c. Trx cell (col 2 in a project with trx) -> enabled.
simulateMainTableSelection(mmc, [1, 2]) ;
assertEnable(mmc.pbChangePath, true, 'trx cell selected') ;

% 1d. Num Labels cell (col 3 with trx) -> disabled.
simulateMainTableSelection(mmc, [1, 3]) ;
assertEnable(mmc.pbChangePath, false, 'num-labels cell selected') ;

% 1e. Selection cleared again -> disabled.
simulateMainTableSelection(mmc, zeros(0, 2)) ;
assertEnable(mmc.pbChangePath, false, 'selection cleared') ;


% --- Section 2: Test that MMC.determineSelectedPathCell_() works -------------------------
simulateMainTableSelection(mmc, [3, 1]) ;
[iMov, iView, isMovie, isValid] = mmc.determineSelectedPathCell_() ;
assertTrue(isValid && iMov == 3 && iView == 1 && isMovie, ...
           sprintf('movie cell: iMov=%d iView=%d isMovie=%d isValid=%d', ...
                   iMov, iView, isMovie, isValid)) ;

simulateMainTableSelection(mmc, [4, 2]) ;
[iMov, iView, isMovie, isValid] = mmc.determineSelectedPathCell_() ;
assertTrue(isValid && iMov == 4 && iView == 1 && ~isMovie, ...
           sprintf('trx cell: iMov=%d iView=%d isMovie=%d isValid=%d', ...
                   iMov, iView, isMovie, isValid)) ;

simulateMainTableSelection(mmc, [4, 3]) ;
[~, ~, ~, isValid] = mmc.determineSelectedPathCell_() ;
assertTrue(~isValid, 'num-labels cell should be invalid') ;


% --- Section 3: Test that Labeler.collectPrefixReplacementCandidates() works for all-literal paths (no macros) --
% Every path in the project shares the same parent prefix.
oldPrefix = '/groups/branson/bransonlab/apt/unittest/alice/data' ;
newPrefix = '/data/relocated' ;
excludePathFull = labeler.movieFilePathFull(1, 1, false) ;  % in actual use, the path that would have just been changed
candidates = ...
  labeler.collectPrefixReplacementCandidates(oldPrefix, newPrefix, excludePathFull) ;

% Build the expected set the dumb way: walk every path and decide
% inclusion ourselves.
expected = expectedCandidatesAllLiteral(labeler, oldPrefix, newPrefix, excludePathFull) ;
assertCandidatesEquivalent(candidates, expected) ;


% --- Section 4: Test collectPrefixReplacementCandidates with macros ----
% Inject a macro that captures the common prefix, then macroize the
% raw paths of the first 3 movies (regular mode).  Those should drop
% out of the candidate list because their raw path now contains "$".
labeler.projMacroAdd('aliceData', oldPrefix) ;

modifiedMovies = movieFilesOriginal ;
macroizedMovieIndices = [1, 2, 3] ;
for k = macroizedMovieIndices
  literalPath = movieFilesOriginal{k, 1} ;
  modifiedMovies{k, 1} = strrep(literalPath, oldPrefix, '$aliceData') ;
end
labeler.movieFilesAll = modifiedMovies ;

% New excluded path: pick one of the *non*-macroized regular movies, so
% we still expect the macroized rows to drop out (rather than being
% trivially excluded as "self").
excludePathFull2 = labeler.movieFilePathFull(5, 1, false) ;
candidatesWhenMacrosPresent = ...
  labeler.collectPrefixReplacementCandidates(oldPrefix, newPrefix, excludePathFull2) ;

for k = 1:numel(candidatesWhenMacrosPresent)
  candidate = candidatesWhenMacrosPresent(k) ;
  if candidate.isMovie && ~candidate.isGT && ismember(candidate.iMov, macroizedMovieIndices)
    error('test_mmc_change_path:macroLeak', ...
          'Macroized path was returned as a candidate: %s', candidate.oldPathFull) ;
  end
end

% The remaining candidates should still match what we expect when the
% macroized rows are filtered out manually.
expectedWhenMacrosPresentProto = ...
  expectedCandidatesAllLiteral(labeler, oldPrefix, newPrefix, excludePathFull2) ;
expectedWhenMacrosPresent = filterOutMacroizedMovies(expectedWhenMacrosPresentProto, macroizedMovieIndices) ;
assertCandidatesEquivalent(candidatesWhenMacrosPresent, expectedWhenMacrosPresent) ;

% Restore literal paths and clear macros for the next section.
labeler.movieFilesAll = movieFilesOriginal ;
labeler.projMacroClear() ;


% --- Section 5: end-to-end relocate + prefix application -----------
% Drive the full flow without the dialog: pick a "new prefix" that's a
% symlink back to the old data dir, so relocateMovieFile (which reads
% movie info) sees real files at the new path.  Skipped on non-Unix.
if ~isunix
  fprintf('Skipping end-to-end relocate section on non-Unix.\n') ;
  return
end
newPrefixSymlink = sprintf('/tmp/test_mmc_change_path_%d', polyfillMatlabProcessID()) ;
[status, msg] = system(sprintf('ln -snf "%s" "%s"', oldPrefix, newPrefixSymlink)) ;
if status ~= 0
  error('test_mmc_change_path:symlink', 'ln -s failed: %s', msg) ;
end
symlinkCleanup = ...
  onCleanup(@()(system(sprintf('rm -f "%s"', newPrefixSymlink)))) ;

changedMovieIndex = 2 ;
oldFull = labeler.movieFilePathFull(changedMovieIndex, 1, false) ;
newFull = strrep(oldFull, oldPrefix, newPrefixSymlink) ;
labeler.relocateMovieFile(changedMovieIndex, 1, false, newFull) ;
assertEqualStrings(labeler.movieFilesAll{changedMovieIndex, 1}, newFull, ...
                   'directly relocated movie did not update') ;

% Exclude key is newFull: the just-relocated cell's *current* path.
candidates2 = ...
  labeler.collectPrefixReplacementCandidates(oldPrefix, newPrefixSymlink, newFull) ;
% The just-relocated cell already has newFull; it should not appear.
for k = 1:numel(candidates2)
  if strcmp(candidates2(k).oldPathFull, newFull)
    error('test_mmc_change_path:relocatedLeak', ...
          'Just-relocated movie appeared as candidate') ;
  end
end
% Sanity: GT row 4 in this project happens to share its file with
% regular row 2 (the row we just changed).  Confirm that GT row 4 is
% nonetheless still a candidate --- a regression check for the bug
% where excluding by old path also dropped sibling cells that
% coincidentally pointed at the same file.
foundGtTwin = false ;
for k = 1:numel(candidates2)
  candidate = candidates2(k) ;
  if candidate.isGT && candidate.isMovie && strcmp(candidate.oldPathFull, oldFull)
    foundGtTwin = true ;
    break
  end
end
if ~foundGtTwin
  error('test_mmc_change_path:gtTwinDropped', ...
        'GT cell sharing the changed cell''s file should still be a candidate') ;
end

% Apply every candidate in one batch.  After this, every path should sit
% under the symlink prefix --- and because the symlink resolves to the
% same data dir, every file still exists.
labeler.relocateFiles(candidates2) ;

% Regular and GT movie paths should now all start with the symlink prefix.
for r = 1:size(labeler.movieFilesAll, 1)
  fullPath = labeler.movieFilePathFull(r, 1, false) ;
  if ~startsWith(fullPath, newPrefixSymlink)
    error('test_mmc_change_path:prefixMissed', ...
          'Movie row %d not under newPrefix: %s', r, fullPath) ;
  end
end
for r = 1:size(labeler.movieFilesAllGT, 1)
  fullPath = labeler.movieFilePathFull(r, 1, true) ;
  if ~startsWith(fullPath, newPrefixSymlink)
    error('test_mmc_change_path:gtPrefixMissed', ...
          'GT movie row %d not under newPrefix: %s', r, fullPath) ;
  end
end

% relocateFiles already synced existence and fired 'update', so the MMC
% styling is current.  All files exist via the symlink, so no cell pink.
assertPinkCellsEquivalent(mmc.tblMain, zeros(0, 2)) ;

% Restore everything so subsequent tests in the suite see a clean slate.
labeler.movieFilesAll = movieFilesOriginal ;
labeler.movieFilesAllGT = movieFilesGTOriginal ;
labeler.trxFilesAll = trxFilesOriginal ;
labeler.trxFilesAllGT = trxFilesGTOriginal ;

end  % function



function simulateMainTableSelection(mmc, selection)
% Simulate the user clicking on tblMain.  Setting Selection
% programmatically does not fire the CellSelectionCallback, so we
% invoke it explicitly.
%
% selection is an Nx2 matrix of [row col] pairs identifying the cells to
% select, matching the shape uitable.Selection takes when SelectionType
% is 'cell'.  Use zeros(0, 2) to clear the selection; pass a single row
% like [3, 1] to select one cell.
mmc.tblMain.Selection = selection ;
mmc.selectionChangedTblMovies([], []) ;
end  % function



function assertEnable(uiHandle, expectation, label)
% Throw if uiHandle.Enable doesn't match expectation.
actual = strcmp(uiHandle.Enable, 'on') ;
if actual ~= expectation
  error('test_mmc_change_path:enable', ...
        'Expected Enable=%d for %s, got %d', expectation, label, actual) ;
end
end  % function



function assertTrue(condition, message)
% Throw with `message` if `condition` is false.
if ~condition
  error('test_mmc_change_path:assertion', '%s', message) ;
end
end  % function



function assertEqualStrings(actual, expected, label)
% Throw with a labeled message if `actual` and `expected` strings differ.
if ~strcmp(actual, expected)
  error('test_mmc_change_path:strings', ...
        '%s: expected "%s" got "%s"', label, expected, actual) ;
end
end  % function



function expected = expectedCandidatesAllLiteral(labeler, oldPrefix, newPrefix, excludePathFull)
% Independent reimplementation of collectPrefixReplacementCandidates
% that we use only as a test oracle.  Walks every (movie/trx) x
% (regular/GT) cell, includes the ones that share oldPrefix and aren't
% the excluded path, and (when called from the all-literal section) we
% know there are no macros to filter out.
expected = struct('iMov', {}, ...
                  'iView', {}, ...
                  'isGT', {}, ...
                  'isMovie', {}, ...
                  'oldPathFull', {}, ...
                  'newPathFull', {}) ;
for isGT = [false, true]
  rowCount = size(labeler.movieFilePathsAllFull(isGT), 1) ;
  for iMov = 1:rowCount
    for iView = 1:labeler.nview
      movRaw = labeler.movieFilePathRaw(iMov, iView, isGT) ;
      if ~isempty(movRaw) && ~contains(movRaw, '$')
        movFull = labeler.movieFilePathFull(iMov, iView, isGT) ;
        if ~strcmp(movFull, excludePathFull) && startsWith(movFull, oldPrefix)
          expected(end+1) = struct(...
            'iMov', iMov, 'iView', iView, 'isGT', isGT, ...
            'isMovie', true, 'oldPathFull', movFull, ...
            'newPathFull', strrep(movFull, oldPrefix, newPrefix)) ;  %#ok<AGROW>
        end
      end
      trxRaw = labeler.trxFilePathRaw(iMov, iView, isGT) ;
      if ~isempty(trxRaw) && ~contains(trxRaw, '$')
        trxFull = labeler.trxFilePathFull(iMov, iView, isGT) ;
        if ~strcmp(trxFull, excludePathFull) && startsWith(trxFull, oldPrefix)
          expected(end+1) = struct(...
            'iMov', iMov, 'iView', iView, 'isGT', isGT, ...
            'isMovie', false, 'oldPathFull', trxFull, ...
            'newPathFull', strrep(trxFull, oldPrefix, newPrefix)) ;  %#ok<AGROW>
        end
      end
    end
  end
end
end  % function



function result = filterOutMacroizedMovies(candidates, macroizedIndices)
% Drop expected entries that correspond to regular-mode movies in
% any of macroizedIndices.
keep = true(1, numel(candidates)) ;
for k = 1:numel(candidates)
  candidate = candidates(k) ;
  if candidate.isMovie && ~candidate.isGT && ismember(candidate.iMov, macroizedIndices)
    keep(k) = false ;
  end
end
result = candidates(keep) ;
end  % function



function assertCandidatesEquivalent(actual, expected)
% Compare two candidate struct arrays as sets keyed on the tuple
% (isGT, isMovie, iMov, iView).  Bodies are also compared.
if numel(actual) ~= numel(expected)
  error('test_mmc_change_path:candidateCount', ...
        'Candidate count: expected %d, got %d', numel(expected), numel(actual)) ;
end
actualKeys = sortKeys(candidateKeys(actual)) ;
expectedKeys = sortKeys(candidateKeys(expected)) ;
if ~isequal(actualKeys, expectedKeys)
  error('test_mmc_change_path:candidateSet', ...
        'Candidate sets differ.\nExpected:\n%s\nActual:\n%s', ...
        formatKeys(expectedKeys), formatKeys(actualKeys)) ;
end
% For each entry, also confirm the path bodies match.
for k = 1:numel(expected)
  e = expected(k) ;
  match = findCandidate(actual, e.isGT, e.isMovie, e.iMov, e.iView) ;
  if ~strcmp(match.oldPathFull, e.oldPathFull) || ...
      ~strcmp(match.newPathFull, e.newPathFull)
    error('test_mmc_change_path:candidateBody', ...
          'Candidate body mismatch for (isGT=%d isMovie=%d iMov=%d iView=%d):\n  expected old=%s new=%s\n  actual   old=%s new=%s', ...
          e.isGT, e.isMovie, e.iMov, e.iView, ...
          e.oldPathFull, e.newPathFull, match.oldPathFull, match.newPathFull) ;
  end
end
end  % function



function keys = candidateKeys(candidates)
% Project a candidate struct array into an Nx4 [isGT isMovie iMov iView]
% matrix, suitable for set comparison via sortrows/isequal.
keys = zeros(numel(candidates), 4) ;
for k = 1:numel(candidates)
  keys(k, :) = [double(candidates(k).isGT), ...
                double(candidates(k).isMovie), ...
                candidates(k).iMov, ...
                candidates(k).iView] ;
end
end  % function


function keys = sortKeys(keys)
% Sort rows of a key matrix lexicographically.
keys = sortrows(keys) ;
end  % function



function s = formatKeys(keys)
% Format a key matrix as one human-readable line per row, for error
% messages.
lines = arrayfun(@(i)(sprintf('  isGT=%d isMovie=%d iMov=%d iView=%d', ...
                              keys(i, 1), keys(i, 2), keys(i, 3), keys(i, 4))), ...
                 (1:size(keys, 1))', 'UniformOutput', false) ;
s = strjoin(lines, newline) ;
end  % function



function result = findCandidate(candidates, isGT, isMovie, iMov, iView)
% Return the candidate whose tuple matches the given coordinates, or
% throw if none does.
for k = 1:numel(candidates)
  c = candidates(k) ;
  if c.isGT == isGT && c.isMovie == isMovie && c.iMov == iMov && c.iView == iView
    result = c ;
    return
  end
end
error('test_mmc_change_path:candidateMissing', ...
      'No candidate matched (isGT=%d isMovie=%d iMov=%d iView=%d)', ...
      isGT, isMovie, iMov, iView) ;
end  % function



function assertPinkCellsEquivalent(table, expected)
% Throw if the set of pink-styled cells in `table` doesn't match `expected`
% (an Nx2 [row col] matrix).
actual = collectPinkCells(table) ;
expected = sortrows(expected) ;
if ~isequal(actual, expected)
  error('test_mmc_change_path:pinkMismatch', ...
        'Pink-cell mismatch.\nExpected:\n%s\nActual:\n%s', ...
        mat2str(expected), mat2str(actual)) ;
end
end  % function



function pinkCells = collectPinkCells(table)
% Return the [row col] indices of every cell currently styled with the
% file-missing pink background, sorted.
pinkRGB = [1.0, 0.85, 0.85] ;
cfg = table.StyleConfigurations ;
pinkCells = zeros(0, 2) ;
for i = 1 : height(cfg)
  target = string(cfg.Target(i)) ;
  if target ~= "cell"
    continue
  end
  bg = cfg.Style(i).BackgroundColor ;
  if ~isequal(bg, pinkRGB)
    continue
  end
  rawIndex = cfg.TargetIndex(i) ;
  if iscell(rawIndex)
    rawIndex = rawIndex{1} ;
  end
  pinkCells = [pinkCells ; rawIndex] ;  %#ok<AGROW>
end
pinkCells = sortrows(pinkCells) ;
end  % function
