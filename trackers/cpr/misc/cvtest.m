function [phisPr,phisPrAll,err,errPerIter]=cvtest(paramfile1,paramfile2,trainresfile,prunePrm0,initlocs,testresfile) %#ok<INUSL>

load(paramfile1);
N = size(phisTr,1); %#ok<NODEF>
% try
%   load(paramfile2,'expidx');
% catch  %#ok<CTCH>
%   expidx = 1:N;
% end

load(trainresfile);
if isstruct(regModel), %#ok<NODEF>
  regModel = {regModel};
end
ncrossvalsets = numel(regModel);

if nargin < 4 || isempty(prunePrm0),
  prunePrm0 = prunePrm;
end
if nargin < 5 || isempty(initlocs),
  initlocs = prunePrm0.initlocs;
end
if nargin < 6,
  testresfile = '';
end

%Test model
phisPr = nan(size(phisTr));
phisPrAll = nan([size(phisTr),prunePrm0.numInit]);

errPerIter = nan(size(phisTr,1),regModel{1}.T+1);

for cvi = 1:ncrossvalsets,
  
  idxtest = cvidx == cvi;
  piT = repmat(permute(initlocs,[3,1,2]),[nnz(idxtest),1,1]);
  bboxesTe = bboxesTr(idxtest,:); %#ok<NODEF>

  [phisPr(idxtest,:),phisPrAll(idxtest,:,:),~,~,p_t] = test_rcpr([],bboxesTe,IsTr(idxtest),regModel{cvi},regPrm,prunePrm0,piT); 
  [errPerEx] = shapeGt('dist',regModel{cvi}.model,phisPr(idxtest,:),phisTr(idxtest,:));
  errcurr = mean(errPerEx);

  [LN,D,T] = size(p_t);
  L = size(initlocs,2);
  N = nnz(idxtest);
  assert(L*N==LN);
  for t = 1:T,
    p_tcurr = permute(reshape(p_t(:,:,t),[N,L,D]),[1,3,2]);
    phisPr_t = rcprTestSelectOutput(p_tcurr,regPrm,prunePrm);
    [errPerEx2] = shapeGt('dist',regModel{cvi}.model,phisPr_t,phisTr(idxtest,:));
    errPerIter(idxtest,t) = errPerEx2;
  end

  fprintf('Error for validation set %d = %f\n',cvi,errcurr);

end

err = mean(errPerIter(:,end));

if ~isempty(testresfile),
  
  save(testresfile,'phisPr','phisPrAll','err','errPerIter','trainresfile','paramfile1','paramfile2');
  
end


