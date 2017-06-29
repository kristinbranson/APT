function changeAPTpaths(oldpathSegment2replace, newpathSegment, dataroot)
%changes all paths in a project if videos have been moved around
%
% oldpathSegment2replace = part of path string that is wrong, only has to
% be part of the path not whole path
%
% newpathSegment = string to replace oldpathSegment2replace with
%
% dataroot = Used if you would like to use path macros in your project's
% video paths
% (https://github.com/kristinbranson/APT/wiki/Path-Macros).  Set this to the
% part of path that will be replaced with a path macro e.g. dataroot = 'Z:'
% if your paths all start with Z: and you would like this to be replaced 
% with the $dataroot macro.  Set to empty (dataroot=[]) otherwise.

[fname,path]=uigetfile(['*.lbl'],'Select APT project to replace path for');


loadFname = [path,fname];

saveFname =loadFname;

lbl = load(loadFname,'-mat');

oldPaths = lbl.movieFilesAll;

%replacing dataroot part of path with $dataroot macro
if ~isempty(dataroot)
    pathsMinusDataRoot = regexprep(oldPaths,dataroot,'$dataroot');
else
    pathsMinusDataRoot = oldPaths;
end

%replacing text segment
newPaths = regexprep(pathsMinusDataRoot,oldpathSegment2replace,newpathSegment);

lbl.movieFilesAll = newPaths;

lbl.projMacros.dataroot = dataroot;

save(saveFname,'-mat','-struct','lbl');