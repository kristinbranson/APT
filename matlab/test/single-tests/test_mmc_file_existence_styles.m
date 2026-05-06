function test_mmc_file_existence_styles()
% Test the file-existence cell coloring in MovieManagerController.
% Open the MMC, verify all cells start white, then break each movie and
% trx file path in turn (by directly mutating the Labeler), and confirm
% that only the corresponding row's Movie or Trx cell is painted pink.
% The Labeler now syncs file existence automatically when the path
% arrays are assigned.

% linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
linux_project_file_path = ...
  '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3_lightly_trained.lbl' ;
if ispc()
  project_file_path = strrep(linux_project_file_path, '/groups/branson/bransonlab', 'Z:') ;
  replace_path = { '/groups/branson/bransonlab', 'Z:' } ;
else
  project_file_path = linux_project_file_path ;
  replace_path = [] ;
end

[labeler, controller] = ...
  StartAPT('projfile', project_file_path, ...
           'replace_path', replace_path) ;
cleaner = onCleanup(@()(delete(controller))) ;  % deletes labeler too
cleaner2 = onCleanup(@()(delete(labeler))) ;  % belt-and-suspenders

% Open the Movie Manager via the menu actuation.
controller.menu_file_managemovies_actuated_([], []) ;
mmc = controller.movieManagerController_ ;
if isempty(mmc) || ~isvalid(mmc.hFig)
  error('test_mmc_file_existence_styles:noMMC', 'Movie Manager did not open') ;
end
uit = mmc.tblMovies ;  % the uitable

% Initially every file exists on disk; no cells should be pink.
assertPinkCells(uit, zeros(0, 2)) ;

% Walk every movie file path; break it, verify, restore.
movieFilesOriginal = labeler.movieFilesAll ;
[rowCount, viewCount] = size(movieFilesOriginal) ;
fakeMoviePath = '/this/path/does/not/exist/foo.avi' ;
for rowIndex = 1 : rowCount
  for viewIndex = 1 : viewCount
    newMovs = movieFilesOriginal ;
    newMovs{rowIndex, viewIndex} = fakeMoviePath ;
    labeler.movieFilesAll = newMovs ;
    pause(0.5) ;  % just so you can see it
    assertPinkCells(uit, [rowIndex, 1]) ;

    labeler.movieFilesAll = movieFilesOriginal ;
    assertPinkCells(uit, zeros(0, 2)) ;
  end
end

% Same for trx files, but only the entries that are non-empty in the
% original project (empty entries don't represent real files and don't
% render an interesting cell).
trxFilesOriginal = labeler.trxFilesAll ;
hasAnyTrx = any(cellfun(@(x)(~isempty(x)), trxFilesOriginal(:))) ;
if hasAnyTrx
  [rowCount, viewCount] = size(trxFilesOriginal) ;
  fakeTrxPath = '/this/path/does/not/exist/foo.trx.mat' ;
  for rowIndex = 1 : rowCount
    for viewIndex = 1 : viewCount
      if isempty(trxFilesOriginal{rowIndex, viewIndex})
        continue
      end
      newTrx = trxFilesOriginal ;
      newTrx{rowIndex, viewIndex} = fakeTrxPath ;
      labeler.trxFilesAll = newTrx ;
      pause(0.5) ;  % just so you can see it
      assertPinkCells(uit, [rowIndex, 2]) ;

      labeler.trxFilesAll = trxFilesOriginal ;
      assertPinkCells(uit, zeros(0, 2)) ;
    end
  end
end
end  % function



function assertPinkCells(uit, expected)
% Throw if the set of cells styled pink in `uit` does not match
% `expected` (an Nx2 [row, col] matrix).
actual = collectPinkCells(uit) ;
expected = sortrows(expected) ;
if ~isequal(actual, expected)
  error('test_mmc_file_existence_styles:wrongStyles', ...
        'Pink-cell mismatch.  Expected:\n%s\nActual:\n%s', ...
        mat2str(expected), mat2str(actual)) ;
end
end  % function



function pinkCells = collectPinkCells(uit)
% Return the [row, col] indices of every cell currently styled with
% the file-missing pink background, sorted.
pinkRGB = [1.0, 0.85, 0.85] ;
cfg = uit.StyleConfigurations ;
pinkCells = zeros(0, 2) ;
for i = 1 : height(cfg)
  target = string(cfg.Target(i)) ;  % robust to cellstr/string/categorical
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
