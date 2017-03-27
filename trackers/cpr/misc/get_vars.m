[file,folder]=uigetfile('*.*');
moviefile=fullfile(folder, file);

labeledpos=reshape(labeledpos,4,[])';
islabeled=~any(isnan(labeledpos),2);
labeledidx=find(islabeled);

phis=labeledpos(islabeled,:);

[readframe,~,fid] = get_readframe_fcn(moviefile);
Is=cell(numel(labeledidx,1));
for i=1:numel(labeledidx)
    Is{i}=rgb2gray(readframe(labeledidx(i)));
end

bboxes=[1 1 fliplr(size(Is{1}))];
bboxes=repmat(bboxes,numel(labeledidx),1);

