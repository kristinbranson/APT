function enumerate_test_suite_data_files(testRootPath, outputPath)
% Enumerate all external files referenced by the test suite's .lbl projects.
%
% testRootPath: path to the test directory tree to scan for .lbl references.
%   Defaults to the sibling 'test/' directory of this file (apt/matlab/test).
% outputPath: path to write the per-project enumeration to.
%   Defaults to 'lbl_referenced_files.txt' in the current directory.
%
% Stage 1: scan every .m file under testRootPath for .lbl path references and
%   produce a deduplicated, sorted list of .lbl project files used by the
%   test suite.  Two heuristics are applied:
%     1. Any single-quoted absolute Linux path ending in .lbl (i.e. matching
%        '/...lbl').  This catches both the canonical /groups/branson/bransonlab/
%        apt/unittest/ projects and ad-hoc paths like the one in
%        test_compare_trackers.m that lives under the user's home directory.
%     2. Calls of the form fullfile(unittest_dir_path, '<name>.lbl'), which the
%        test helpers use; <name>.lbl is resolved against the Linux unittest dir
%        returned by get_test_project_paths() on Linux.
%
% Stage 2: for each .lbl, load the project with a (batch-mode) Labeler and
%   read the macro-resolved paths from movieFilesAllFull, movieFilesAllGTFull,
%   trxFilesAllFull, and trxFilesAllGTFull.  The output file lists, per .lbl,
%   the referenced files grouped by category, and ends with a deduplicated
%   global list of every unique file path across all .lbl files.

  if ~exist('testRootPath', 'var') || isempty(testRootPath)
    thisFileDir = fileparts(mfilename('fullpath')) ;
    matlabDir = fileparts(thisFileDir) ;
    testRootPath = fullfile(matlabDir, 'test') ;
  end
  if ~exist('outputPath', 'var') || isempty(outputPath)
    outputPath = 'lbl_referenced_files.txt' ;
  end

  % Stage 1: discover .lbl files referenced by the test suite
  lblPaths = discoverLblFilesFromTestSuite_(testRootPath) ;
  fprintf('Discovered %d .lbl file(s) referenced in test suite under %s.\n', ...
          numel(lblPaths), testRootPath) ;
  for printIndex = 1 : numel(lblPaths)
    fprintf('  %s\n', lblPaths{printIndex}) ;
  end

  % For testing: only look at the first few .lbl files.
  testingLimit = 3 ;
  if numel(lblPaths) > testingLimit
    lblPaths = lblPaths(1 : testingLimit) ;
  end
  lblCount = numel(lblPaths) ;
  fprintf('Processing %d .lbl file(s).\n', lblCount) ;

  % Open the output file
  outFid = fopen(outputPath, 'w') ;
  if outFid < 0
    error('Could not open output file ''%s'' for writing.', outputPath) ;
  end
  outCleaner = onCleanup(@()(fclose(outFid))) ;  %#ok<NASGU>

  allPaths = {} ;
  failures = {} ;

  for lblIndex = 1 : lblCount
    lblPath = lblPaths{lblIndex} ;
    fprintf('\n[%d/%d] %s\n', lblIndex, lblCount, lblPath) ;
    fprintf(outFid, '================================================================\n') ;
    fprintf(outFid, 'PROJECT: %s\n', lblPath) ;
    fprintf(outFid, '================================================================\n') ;

    if ~exist(lblPath, 'file')
      fprintf(outFid, '  <FILE NOT FOUND>\n\n') ;
      failures{end+1, 1} = sprintf('%s (file not found)', lblPath) ;  %#ok<AGROW>
      continue
    end

    try
      paths = enumerateOneLbl_(lblPath, outFid) ;
      allPaths = [allPaths ; paths(:)] ;  %#ok<AGROW>
    catch loadError
      fprintf(outFid, '  <ERROR LOADING: %s>\n\n', loadError.message) ;
      failures{end+1, 1} = sprintf('%s (%s)', lblPath, loadError.message) ;  %#ok<AGROW>
    end
    fprintf(outFid, '\n') ;
  end

  % Deduplicated, sorted global list
  uniquePaths = unique(allPaths) ;
  fprintf(outFid, '================================================================\n') ;
  fprintf(outFid, 'ALL UNIQUE REFERENCED FILES (%d total)\n', numel(uniquePaths)) ;
  fprintf(outFid, '================================================================\n') ;
  for pathIndex = 1 : numel(uniquePaths)
    fprintf(outFid, '%s\n', uniquePaths{pathIndex}) ;
  end

  if ~isempty(failures)
    fprintf(outFid, '\n================================================================\n') ;
    fprintf(outFid, 'FAILURES (%d)\n', numel(failures)) ;
    fprintf(outFid, '================================================================\n') ;
    for failureIndex = 1 : numel(failures)
      fprintf(outFid, '%s\n', failures{failureIndex}) ;
    end
  end

  fprintf('\nDone.  Wrote %d unique paths to %s.\n', numel(uniquePaths), outputPath) ;
  if ~isempty(failures)
    fprintf('%d project(s) failed; see end of output file.\n', numel(failures)) ;
  end
end  % function


function lblPaths = discoverLblFilesFromTestSuite_(testRootPath)
  % Scan all .m files under testRootPath and extract .lbl file paths
  % referenced in the source.
  if ~exist(testRootPath, 'dir')
    error('Test root directory ''%s'' does not exist.', testRootPath) ;
  end

  unittestDirOnLinux = '/groups/branson/bransonlab/apt/unittest' ;

  % Recursively list all .m files in the test tree
  mFileInfos = dir(fullfile(testRootPath, '**', '*.m')) ;

  absolutePattern = '''(/[^'']+\.lbl)''' ;
  fullfilePattern = 'fullfile\s*\(\s*unittest_dir_path\s*,\s*''([^'']+\.lbl)''' ;

  discovered = {} ;
  for fileIndex = 1 : numel(mFileInfos)
    info = mFileInfos(fileIndex) ;
    fullPath = fullfile(info.folder, info.name) ;
    text = fileread(fullPath) ;

    absoluteTokens = regexp(text, absolutePattern, 'tokens') ;
    for tokIndex = 1 : numel(absoluteTokens)
      discovered{end+1, 1} = absoluteTokens{tokIndex}{1} ;  %#ok<AGROW>
    end

    fullfileTokens = regexp(text, fullfilePattern, 'tokens') ;
    for tokIndex = 1 : numel(fullfileTokens)
      discovered{end+1, 1} = ...
        [unittestDirOnLinux '/' fullfileTokens{tokIndex}{1}] ;  %#ok<AGROW>
    end
  end

  lblPaths = unique(discovered) ;
end  % function


function paths = enumerateOneLbl_(lblPath, outFid)
  % Load a single .lbl and emit the referenced file paths to outFid.
  % Returns a column cellstr of all referenced paths (non-empty).
  labeler = Labeler('isInBatchMode', true) ;
  cleaner = onCleanup(@()(delete(labeler))) ;  %#ok<NASGU>

  labeler.projLoad(lblPath, 'nomovie', true) ;

  movieFiles = flattenCellstr_(labeler.movieFilesAllFull) ;
  movieGTFiles = flattenCellstr_(labeler.movieFilesAllGTFull) ;
  trxFiles = flattenCellstr_(labeler.trxFilesAllFull) ;
  trxGTFiles = flattenCellstr_(labeler.trxFilesAllGTFull) ;

  writeSection_(outFid, 'Movies (regular)', movieFiles) ;
  writeSection_(outFid, 'Movies (GT)', movieGTFiles) ;
  writeSection_(outFid, 'Trx files (regular)', trxFiles) ;
  writeSection_(outFid, 'Trx files (GT)', trxGTFiles) ;

  paths = [movieFiles ; movieGTFiles ; trxFiles ; trxGTFiles] ;
end  % function


function flat = flattenCellstr_(cellOfStr)
  % Flatten a cell array of char to a column cellstr with empty entries removed.
  if isempty(cellOfStr)
    flat = cell(0, 1) ;
    return
  end
  flat = reshape(cellOfStr, [], 1) ;
  isKeeper = ~cellfun(@isempty, flat) ;
  flat = flat(isKeeper) ;
end  % function


function writeSection_(outFid, label, paths)
  % Write a labeled list of paths to outFid.
  fprintf(outFid, '  %s (%d):\n', label, numel(paths)) ;
  if isempty(paths)
    fprintf(outFid, '    (none)\n') ;
    return
  end
  for pathIndex = 1 : numel(paths)
    fprintf(outFid, '    %s\n', paths{pathIndex}) ;
  end
end  % function
