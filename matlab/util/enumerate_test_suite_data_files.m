function enumerate_test_suite_data_files(testRootPath, outputPath)
% Enumerate all external files referenced by the test suite's .lbl projects.
%
% testRootPath: path to the test directory tree to scan for .lbl references.
%   Defaults to the sibling 'test/' directory of this file (apt/matlab/test).
% outputPath: path to write the per-project enumeration to.
%   Defaults to 'lbl_referenced_files.txt' in the current directory.
%
% Stage 1: scan every .m file under testRootPath for assignments of the form
%   linux_project_file_path = '<path>.lbl' (optionally split across two lines
%   with a MATLAB continuation '...').  This is the project-wide convention
%   for naming the .lbl that a test will load via localize_test_project_path,
%   so it identifies exactly the tests that read a .lbl file.
%
% Stage 2: for each .lbl, load the project with a (batch-mode) Labeler and
%   read the macro-resolved paths from movieFilesAllFull, movieFilesAllGTFull,
%   trxFilesAllFull, and trxFilesAllGTFull.  The output file lists each .lbl
%   path on its own line at the left margin, followed by the files it refers
%   to, each indented by two spaces.

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
  testingLimit = inf ;
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
  outCleaner = onCleanup(@()(fclose(outFid))) ;  

  for lblIndex = 1 : lblCount
    lblPath = lblPaths{lblIndex} ;
    fprintf('\n[%d/%d] %s\n', lblIndex, lblCount, lblPath) ;
    fprintf(outFid, '%s\n', lblPath) ;

    if ~exist(lblPath, 'file')
      continue
    end

    try
      referencedPaths = enumerateOneLbl_(lblPath) ;
      for refIndex = 1 : numel(referencedPaths)
        fprintf(outFid, '  %s\n', referencedPaths{refIndex}) ;
      end
    catch loadError
      fprintf('Error loading %s: %s\n', lblPath, loadError.message) ;
    end
  end

  fprintf('\nDone.  Wrote results to %s.\n', outputPath) ;
end  % function


function lblPaths = discoverLblFilesFromTestSuite_(testRootPath)
  % Scan all .m files under testRootPath and extract the .lbl path from
  % each linux_project_file_path = '<path>.lbl' assignment.  The MATLAB
  % line-continuation '...' between the '=' and the opening quote is
  % tolerated so that both single-line and split-line forms are matched.
  if ~exist(testRootPath, 'dir')
    error('Test root directory ''%s'' does not exist.', testRootPath) ;
  end

  % Recursively list all .m files in the test tree
  mFileInfos = dir(fullfile(testRootPath, '**', '*.m')) ;

  assignmentPattern = 'linux_project_file_path\s*=\s*(?:\.\.\.\s*)?''([^'']+\.lbl)''' ;

  discovered = {} ;
  for fileIndex = 1 : numel(mFileInfos)
    info = mFileInfos(fileIndex) ;
    fullPath = fullfile(info.folder, info.name) ;
    text = fileread(fullPath) ;

    tokens = regexp(text, assignmentPattern, 'tokens') ;
    for tokIndex = 1 : numel(tokens)
      discovered{end+1, 1} = tokens{tokIndex}{1} ;  %#ok<AGROW>
    end
  end

  lblPaths = unique(discovered) ;
end  % function


function paths = enumerateOneLbl_(lblPath)
  % Load a single .lbl and return all referenced file paths as a column
  % cellstr (non-empty entries only).
  labeler = Labeler('isInBatchMode', true) ;
  cleaner = onCleanup(@()(delete(labeler))) ;  

  labeler.projLoad(lblPath, 'nomovie', true) ;

  movieFiles = flattenCellstr_(labeler.movieFilesAllFull) ;
  movieGTFiles = flattenCellstr_(labeler.movieFilesAllGTFull) ;
  trxFiles = flattenCellstr_(labeler.trxFilesAllFull) ;
  trxGTFiles = flattenCellstr_(labeler.trxFilesAllGTFull) ;

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
