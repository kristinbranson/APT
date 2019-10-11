% Convert labeler output data to tracker input data.
%   - The loaded file will be a text file ocntaining the path to one or
%   several label files.
%   - Each labels file will be the output of the Labeler, containing:
%      + expdirs,  a cell containing the path to each video
%      + labeledpos_perexp, a cell array containing the labels for each
%      video (one cell per video). Inside ieach cell, a nPoints x 2 x
%      nFrames matrix.
%   - The output variables will be stored in two files:
%      + output_file.mat (output_file is the name selected by the user):
%      labels (phisTr), bounding boxes (bboxesTr, in this case is just the
%      dimensions of the figure), video locations (expdirs_all) and
%      correspondence between each label and its video (phis2dir).
%      +  output_file_Is.mat: labeled frames (IsTr, cell arrar fith nFrames)

clear

[file,folder]=uigetfile('*.txt');
expfile=fullfile(folder,file);
%%
fid = fopen(expfile,'r');
labelfile={};
while true
    l = fgetl(fid);
    if ~ischar(l),
        break;
    end
    l = strtrim(l);
    labelfile{end+1}=l;
end
fclose(fid);

phisTr=[];
IsTr={};
expdirs_all=[];
phis2dir=[];
N=1;
for i=1:numel(labelfile)
    load(labelfile{i})
    if isunix
        my_driver='/tier2/hantman/';
    else
        my_driver=labelfile{i}(1:3);
    end
    their_driver=expdirs{1}(1:3);
    expdirs=strrep(expdirs,their_driver,my_driver);
    if isunix
        expdirs=strrep(expdirs,'\','/');
    end
    expdirs_all=[expdirs_all,expdirs];
    moviefile=fullfile(expdirs,'movie_comb.avi');
    for j=1:numel(moviefile)
        labeledpos=labeledpos_perexp{j};
        labeledpos=reshape(labeledpos,4,[])';
        islabeled=~any(isnan(labeledpos),2);
        labeledidx=find(islabeled);
        phisTr=[phisTr;labeledpos(islabeled,:)];
        phis2dir=[phis2dir;N*ones(sum(islabeled),1)];
        disp(length(phisTr));
        [readframe,~,fidm] = get_readframe_fcn(moviefile{j});
        for k=1:numel(labeledidx)
            IsTr=[IsTr;rgb2gray_cond(readframe(labeledidx(k)))];
        end
        
        if fidm>0
            fclose(fidm);
        end
        N=N+1;
    end
end
bboxesTr=[1 1 fliplr(size(IsTr{1}))];
bboxesTr=repmat(bboxesTr,numel(IsTr),1);

[Sfile,Sfolder] = uiputfile('*.mat');
phisfile = fullfile(Sfolder,Sfile);
Isfile = fullfile(Sfolder,[Sfile(1:end-4),'_Is.mat']);
save(phisfile,'phisTr','bboxesTr','expdirs_all','phis2dir')
save(Isfile,'IsTr','-v7.3')