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
% Currently this input can only be used if the project doesn't have a path macro
% already.


[fname,path]=uigetfile(['*.lbl'],'Select APT project to replace path for');


loadFname = [path,fname];

saveFname =loadFname;

lbl = load(loadFname,'-mat');

% Catching situation where user requests a path macro and another one already exists in project. 
% to do: alter this code to be able to deal with this.
if isfield(lbl.projMacros,'dataroot') && ~isempty(dataroot) 
   error('Project already has path macro.  This code can''t currently deal with replacing an old macro with a new one.  Use Gui to replace path macro then re-run with dataroot input = []')
end

oldPaths = lbl.movieFilesAll;

%replacing dataroot part of path with $dataroot macro
if ~isempty(dataroot)
    pathsMinusDataRoot = regexprep(oldPaths,dataroot,'$dataroot');
else
    pathsMinusDataRoot = oldPaths;
end

%replacing text segment
newPaths = strrep(pathsMinusDataRoot,oldpathSegment2replace,newpathSegment);

lbl.movieFilesAll = newPaths;

if ~isempty(dataroot)
    lbl.projMacros.dataroot = dataroot;
end

save(saveFname,'-mat','-struct','lbl');