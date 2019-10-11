% Convert labeler output data (including occlusion) to tracker input data.
%   - The loaded files (one per view) will be the output of the Labeler, containing:
%      + moviefile,  a text string containing the file location
%      + labeledpo, a nPoints x 3 x nFrames matrix containing the labaler
%      cooridnates and their occlusion state. 
%   - The output variables will be stored in two files:
%      + output_file.mat (output_file is the name selected by the user):
%      labels (phisTr, cell array with the labels for each view), bounding
%      boxes (bboxesTr, cell array with the labels for each view containing
%      just the dimensions of the figure), video location  (moviefile).
%      +  output_file_Is.mat: labeled frames (IsTr, cell array with nFrames
%      x nViews)

clear

[files,folder]=uigetfile('*.mat','MultiSelect','on');
files=sort(files);
nviews=numel(files);
labeledpos=cell(1,nviews);
moviefile=cell(1,nviews);
islabeled=nan(1,nviews);
phisTr=cell(1,nviews);
bboxesTr=cell(1,nviews);
for i=1:nviews
    expfile=fullfile(folder,files{i});
    %%
    data=load(expfile);
    labeledpos{i}=data.labeledpos;
    moviefile{i}=data.moviefile;
    D=size(labeledpos{i},1)*size(labeledpos{i},2);
    labeledpos{i}=reshape(labeledpos{i},D,[])';
    nframes=size(labeledpos{i},1);
    islabeled(1:nframes,i)=~any(isnan(labeledpos{i}),2);
end
    islabeled=all(islabeled,2);
    labeledidx=find(islabeled);
    nfl=numel(labeledidx);
    np=D/3;
    allpos=nan(nfl*np,nviews*2);
    isocluded=false(nfl*np,1);
    IsTr=cell(nfl,nviews);
for i=1:nviews
    phisTr{i}=labeledpos{i}(islabeled,:);

    [readframe,~,fidm] = get_readframe_fcn(moviefile{i});
    for k=1:nfl
        allpos(np*(k-1)+1:np*k,2*(i-1)+1:2*i)=reshape(phisTr{i}(k,1:2*np),np,2);
        isocluded(np*(k-1)+1:np*k,1)=isocluded(np*(k-1)+1:np*k,1)|reshape(phisTr{i}(k,2*np+1:3*np),np,1);
        IsTr{k,i}=rgb2gray_cond(readframe(labeledidx(k)));
    end

    if fidm>0
        fclose(fidm);
    end
    bboxes1=[1 1 fliplr(size(IsTr{1}))];
    bboxesTr{i}=repmat(bboxes1,numel(IsTr),1);
end
if numel(phisTr)==1
    phisTr  =phisTr{i};
    bboxesTr = bboxesTr{i};
end
[Sfile,Sfolder] = uiputfile('*.mat');
phisfile = fullfile(Sfolder,Sfile);
Isfile = fullfile(Sfolder,[Sfile(1:end-4),'_Is.mat']);
save(phisfile,'phisTr','bboxesTr','moviefile')
save(Isfile,'IsTr','-v7.3')
