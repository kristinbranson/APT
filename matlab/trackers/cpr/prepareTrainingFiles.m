function prepareTrainingFiles(infile,outfile,exps4train,nperexp)

if nargin<4,
  nperexp = inf;
end

O = load(infile);

if nargin < 3,
  exps4train = 1:numel(O.expdirs);
end

phisTr=[];
IsTr={};
phis2dir=[];
N=1;
moviefile=fullfile(O.expdirs,'movie_comb.avi');
sz = size(O.labeledpos_perexp{1});
npts = prod(sz(1:2));
expdirs_all = O.expdirs(exps4train);
for j=exps4train(:)'
  labeledpos=O.labeledpos_perexp{j};
  labeledpos=reshape(labeledpos,npts,[])';
  islabeled=~any(isnan(labeledpos),2);
  labeledidx=find(islabeled);
  if numel(labeledidx)>nperexp,
    labeledidx = randsample(labeledidx,nperexp);
    islabeled = false(size(islabeled));
    islabeled(labeledidx) = true;
  end  
  
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
bboxesTr=[1 1 fliplr(size(IsTr{1}))];
bboxesTr=repmat(bboxesTr,numel(IsTr),1);

if strcmp(outfile(end-3:end),'.mat'),
  outfile = outfile(1:end-4);
end
Isfile = [outfile '_Is.mat'];
save([outfile '.mat'],'phisTr','bboxesTr','expdirs_all','phis2dir')
save(Isfile,'IsTr','-v7.3')