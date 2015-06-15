folder = uigetdir;
content = dir(folder);
nfiles = numel(content)-2;
p_all = cell(nfiles,1);
moviefiles_all = cell(1,nfiles);
for i=1:nfiles
    file = content(i+2).name;
    load(fullfile(folder,file));
    p_all{i} = p_med_bb;
    moviefiles_all{i} = params.moviefile;
end
params = rmfield(params,'moviefile');
    