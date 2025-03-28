projectFileName = ...
  '/groups/branson/bransonlab/apt/unittest/alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019.lbl' ;

[labeler, controller] = StartAPT('projfile', projectFileName, 'isInDebugMode', true,  'isInAwsDebugMode', true, 'isInYodaMode', true) ;

%%
pathFromRegularMovieIndex = labeler.movieFilesAllFull ;
pathFromGTMovieIndex = labeler.movieFilesAllGTFull ;

%delete(controller) ;

pathFromMovieIndex = vertcat(pathFromRegularMovieIndex, pathFromGTMovieIndex) ;

newDataRoot = '/groups/branson/bransonlab/apt/unittest/alice/data' ;
system_with_error_handling(sprintf('mkdir -p ''%s''', newDataRoot)) ;

function copyMovieDir(moviePath, newDataRoot)
  sourceExpDirPath = fileparts2(moviePath) ;
  [~,sourceExpDirName] = fileparts2(sourceExpDirPath) ;
  destExpDirPath = fullfile(newDataRoot, sourceExpDirName) ;
  system_with_error_handling(sprintf('mkdir -p ''%s''', destExpDirPath)) ;
  % movie
  sourceMoviePath = fullfile(sourceExpDirPath, 'movie.ufmf') ;
  destMoviePath = fullfile(destExpDirPath, 'movie.ufmf') ;
  command = sprintf('rsync --inplace ''%s'' ''%s''', sourceMoviePath, destMoviePath) ;  
  system_with_error_handling(command) ;
  % registered_trx.mat
  sourceMoviePath = fullfile(sourceExpDirPath, 'registered_trx.mat') ;
  destMoviePath = fullfile(destExpDirPath, 'registered_trx.mat') ;
  command = sprintf('rsync --inplace ''%s'' ''%s''', sourceMoviePath, destMoviePath) ;  
  system_with_error_handling(command) ;  
end

cellfun(@(moviePath)(copyMovieDir(moviePath, newDataRoot)), pathFromMovieIndex) ;

%labeler.projMacros.dataroot = newDataRoot ;
labeler.projMacroSet('dataroot', newDataRoot)
