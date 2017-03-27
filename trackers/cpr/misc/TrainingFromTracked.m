nframessample = 100;

nvids = numel(moviefiles_all);
ninfo = numel(importantframes);

if isunix
    old_drive = 'Y:\';
    new_drive = '/tier2/hantman/';
    moviefiles_all=strrep(moviefiles_all,old_drive,new_drive);
    moviefiles_all=strrep(moviefiles_all,'\','/');
else
    old_drive = '/tier2/hantman/';
    new_drive = 'Y:\';
    moviefiles_all=strrep(moviefiles_all,old_drive,new_drive);
    moviefiles_all=strrep(moviefiles_all,'/','\');
end

phisTr = [];
phis2dirTr = [];
moviefiles_allTr = [];
IsTr = [];
for i=1:ninfo
    info_file = [importantframes(i).expdir,'/movie_comb.avi'];
    if isunix
        info_file=strrep(info_file,old_drive,new_drive);
        info_file=strrep(info_file,'\','/');
    else
        info_file=strrep(info_file,old_drive,new_drive);
        info_file=strrep(info_file,'/','\');
    end
    isfile = find(strcmp(moviefiles_all,info_file));
    if ~isempty(isfile)
        nframes = size(p_all{isfile},1);
        first = max(1,importantframes(i).firstbehavior-10);
        last = min(nframes,importantframes(i).lastbehavior+10);
        nframessample = last-first+1;
        framessample = unique(round(linspace(first,last,nframessample)));
        nframesS = numel(framessample);
        phisTr = [phisTr;p_all{i}(framessample,:)];
        phis2dirTr = [phis2dirTr;i*ones(nframesS,1)];
        moviefiles_allTr =  [moviefiles_allTr moviefiles_all(isfile)];
        [readframe,~,fidm] = get_readframe_fcn(moviefiles_all{isfile});
        Is = cell(nframesS,1);
        for j=1:nframesS
            Is{j}=rgb2gray_cond(readframe(framessample(j)));
        end
        IsTr = [IsTr;Is];
    end
end
bboxesTr = repmat([1 1 fliplr(size(Is{end}))],[size(phisTr,1),1]);