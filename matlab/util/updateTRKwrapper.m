function updateTRKwrapper(path,fnames)
%wrapper around importTrkResave.m.  updates .trk tracking info for all APT
%projects selected by user
%
% path, fnames = strings outputting by
%[fnames,path]= uigetfile([hustonlab,'\flp-chrimson_experiments\APT_projectFiles\*.lbl'],'multiselect','on')
% and selecting APT project files to analyze.  Can leave empty and program
% will prompt you to make them using gui.


if isempty(fnames)
    [fnames,path]=uigetfile(['*.lbl'],'Select all APT projects to update tracking for','multiselect','on');
end

%intitialising labeler
lObj = Labeler;


if iscell(fnames)
    nLoops = length(fnames);
else
    nLoops =1;
end

for p = 1:nLoops %for each APT project
    
    try
    
        if iscell(fnames)
            importTrkResave(lObj,[path,fnames{p}])
        else
            importTrkResave(lObj,[path,fnames])
        end

    catch
        lasterr
        disp(['Unable to add new trk files to project:' , fnames{p}])
    end
    
end

close all