function j = GetLabelMovieIdx(ddmoviefiles,lbld,ddrootdir)

view = 1;

if nargin <= 2,
  ddrootdir = '/groups/branson/bransonlab/mayank/stephenCV/';
end
lenddrootdir = numel(ddrootdir);

ddmoviestr = ddmoviefiles{view}(lenddrootdir+1:end);
lenddmoviestr = numel(ddmoviestr);
didfind = false;
for j = 1:size(lbld.movieFilesAll,1),
  if numel(lbld.movieFilesAll{j,view}) >= lenddmoviestr && strcmp(lbld.movieFilesAll{j,view}(end-lenddmoviestr+1:end),ddmoviestr),
    didfind = true;
    break;
  end
end
if ~didfind,
  j = [];
end