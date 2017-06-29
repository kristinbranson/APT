function changeAPTpaths(oldpathSegment2replace, newpathSegment)
%changes all paths in a project if videos have been moved around
%
% oldpathSegment2replace = part of path string that is wrong, only has to
% be part of the path not whole path
%
% newpathSegment = string to replace oldpathSegment2replace with

[fname,path]=uigetfile(['*.lbl'],'Select APT project to replace path for');



dataroot =  [hustonlab,'\']; %root of paths, usually path to hustonlab AKA 'Z:'

loadFname = [path,fname];

%replace this with saveFname = loadFname when working
saveFname =loadFname;

lbl = load(loadFname,'-mat');

oldPaths = lbl.movieFilesAll;

%replacing part of path with data root
pathsMinusDataRoot = regexprep(oldPaths,dataroot,'$dataroot');

%replacing text segment
newPaths = regexprep(pathsMinusDataRoot,oldpathSegment2replace,newpathSegment);

lbl.movieFilesAll = newPaths;

lbl.projMacros.dataroot = dataroot;

save(saveFname,'-mat','-struct','lbl');