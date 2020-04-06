% Convert labeler output data to tracker input data (single video).
%   - The loaded file will be the output of the Labeler, containing:
%      + moviefile,  a text string containing the file location
%      + labeledpos, a nPoints x 2 x nFrames matrix containing the labaler
%   - The output variables will be stored in two files:
%      + output_file.mat (output_file is the name selected by the user):
%      labels (phisTr), bounding boxes (bboxesTr, in this case is just the
%      dimensions of the figure), video location (moviefile).  
%      +  output_file_Is.mat: labeled frames (IsTr, cell array with nFrames)
clear

[file,folder]=uigetfile('*.mat');
expfile=fullfile(folder,file);
%%
load(expfile)
D=size(labeledpos,1)*size(labeledpos,2);
labeledpos=reshape(labeledpos,D,[])';
islabeled=~any(isnan(labeledpos),2);
labeledidx=find(islabeled);
phis=labeledpos(islabeled,:);

[readframe,~,fidm] = get_readframe_fcn(moviefile);
Is=cell(numel(labeledidx),1);
for k=1:numel(labeledidx)
    Is{k}=rgb2gray_cond(readframe(labeledidx(k)));
end

if fidm>0
    fclose(fidm);
end
bboxes=[1 1 fliplr(size(Is{1}))];
bboxes=repmat(bboxes,numel(Is),1);


