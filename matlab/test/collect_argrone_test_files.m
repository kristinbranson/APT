projectFileName = ...
  '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved.lbl' ;

[labeler, controller] = StartAPT('projfile', projectFileName, 'isInDebugMode', true,  'isInAwsDebugMode', true, 'isInYodaMode', true) ;

%%
pathFromRegularMovieIndex = labeler.movieFilesAllFull ;
pathFromGTMovieIndex = labeler.movieFilesAllGTFull ;

%delete(controller) ;

pathFromMovieIndex = vertcat(pathFromRegularMovieIndex, pathFromGTMovieIndex) ;

newDataRoot = '/groups/branson/bransonlab/apt/unittest/alice/data' ;
system_with_error_handling(sprintf('mkdir -p ''%s''', newDataRoot)) ;

% cellfun(@(moviePath)(copyMovieDir(moviePath, newDataRoot)), pathFromMovieIndex) ;
for i = 1 : numel(pathFromMovieIndex) ,
  sourceMoviePath = pathFromMovieIndex{i} ;
  copyMovieDir(sourceMoviePath, newDataRoot) ;
end

function copyMovieDir(sourceMoviePath, newDataRoot)
  sourceExpDirPath = fileparts2(sourceMoviePath) ;
  [~,sourceExpDirName] = fileparts2(sourceExpDirPath) ;
  if strcmp(sourceExpDirName, 'nochr_TrpA71G01_Unknown_RigD_20201212T162439') 
    nop() ;
  end
  destExpDirPath = fullfile(newDataRoot, sourceExpDirName) ;
  system_with_error_handling(sprintf('mkdir -p ''%s''', destExpDirPath)) ;
  % movie
  sourceMoviePath = fullfile(sourceExpDirPath, 'movie.ufmf') ;
  destMoviePath = fullfile(destExpDirPath, 'movie.ufmf') ;
  command = sprintf('rsync -L --inplace ''%s'' ''%s''', sourceMoviePath, destMoviePath) ;  
  system_with_error_handling(command) ;
  % registered_trx.mat
  sourceTrxPath = fullfile(sourceExpDirPath, 'registered_trx.mat') ;
  destTrxPath = fullfile(destExpDirPath, 'registered_trx.mat') ;
  command = sprintf('rsync -L --inplace ''%s'' ''%s''', sourceTrxPath, destTrxPath) ;  
  system_with_error_handling(command) ;  
end

% %%
% function result = moved_file_path(sourceFilePath, newDataRoot)
%   [sourceExpDirPath,sourceFileName] = fileparts2(sourceFilePath) ;
%   [~,sourceExpDirName] = fileparts2(sourceExpDirPath) ;
%   destExpDirPath = fullfile(newDataRoot, sourceExpDirName) ;
%   result = fullfile(destExpDirPath, sourceFileName) ;  
% end
% 
% for i = 1 : numel(labeler.movieFilesAll) ,
%   sourceFilePath = labeler.movieFilesAll{i} ;
%   destFilePath = moved_file_path(sourceFilePath, newDataRoot) ;
%   labeler.movieFilesAll{i} = destFilePath ;
% end
% 
% %%
% for i = 1 : numel(labeler.movieFilesAllGT) ,
%   sourceFilePath = labeler.movieFilesAllGT{i} ;
%   destFilePath = moved_file_path(sourceFilePath, newDataRoot) ;
%   labeler.movieFilesAllGT{i} = destFilePath ;
% end
% 
% %%
% for i = 1 : numel(labeler.trxFilesAll) ,
%   sourceFilePath = labeler.trxFilesAll{i} ;
%   destFilePath = moved_file_path(sourceFilePath, newDataRoot) ;
%   labeler.trxFilesAll{i} = destFilePath ;
% end
% 
% for i = 1 : numel(labeler.trxFilesAllGT) ,
%   sourceFilePath = labeler.trxFilesAllGT{i} ;
%   destFilePath = moved_file_path(sourceFilePath, newDataRoot) ;
%   labeler.trxFilesAllGT{i} = destFilePath ;
% end
