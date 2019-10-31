% Read list of labeled experiments and extract labels.
%   - fid is the file identifer of a previously opened text flie containing
%   the path to one or several label files. 
%   - Each labels file will be the output of the Labeler, containing:
%      + expdirs,  a cell containing the path to each video
%      + labeledpos_perexp, a cell array containing the labels for each
%      video (one cell per video). Inside ieach cell, a nPoints x 2 x
%      nFrames matrix.
%   - If dotest=true, the labels will be output.

function [expdirs_all,moviefiles_all,labeledpos]=read_exp_list_labeled(fid,dotest)
labeldir={};
while true
    l = fgetl(fid);
    if ~ischar(l),
        break;
    end
    l = strtrim(l);
    labeldir{end+1}=l;
end
fclose(fid);

expdirs_all=[];
moviefiles_all=[];
labeledpos=[];
for i=1:numel(labeldir)
    load(labeldir{i},'expdirs','labeledpos_perexp')
    if isunix
        my_driver='/tier2/hantman/';
    else
        my_driver=labeldir{i}(1:3);
    end
    their_driver=expdirs{1}(1:3);
    expdirs=strrep(expdirs,their_driver,my_driver);
    if isunix
        expdirs=strrep(expdirs,'\','/');
    end
    expdirs_all=[expdirs_all expdirs];
    moviefiles_all=[moviefiles_all fullfile(expdirs,'movie_comb.avi')];
    if dotest
        labeledpos=[labeledpos labeledpos_perexp];
    end
end



