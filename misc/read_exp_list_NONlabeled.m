% Read list of NON labeled experiments.
%   - fid is the file identifer of a previously opened text flie containing
%   the path to one or directories containing experiment folders. 
function [expdirs_all,moviefiles_all,labeledpos,dotest]=read_exp_list_NONlabeled(fid)
datadir={};
while true
    l = fgetl(fid);
    if ~ischar(l),
        break;
    end
    l = strtrim(l);
    datadir{end+1}=l;
end
fclose(fid);

expdirs_all=[];
moviefiles_all=[];
labeledpos=[];
dotest=false;
for i=1:numel(datadir)
    content=dir(datadir{i});
    aregood=cellfun(@isempty,strfind({content.name},'.'));
    aredirs=[content.isdir]&aregood;
    expdirs={content(aredirs).name};
    hasmovie=cellfun(@(x) exist(fullfile(datadir{i},x,'movie_comb.avi'),'file'),expdirs)>0;
    expdirs_all=[expdirs_all,cellfun(@(x) fullfile(datadir{i},x),expdirs(hasmovie),'Uni',0)];
    moviefiles_all=[moviefiles_all,cellfun(@(x) fullfile(datadir{i},x,'movie_comb.avi'),expdirs(hasmovie),'Uni',0)];
end



