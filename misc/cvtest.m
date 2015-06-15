function [phisPr,phisPrAll,err]=cvtest(paramfile1,paramfile2,trainresfile,prunePrm0,initlocs,testresfile)

load(paramfile1);
N = size(phisTr,1); %#ok<NODEF>
try
  load(paramfile2,'expidx');
catch  %#ok<CTCH>
  expidx = 1:N;
end

load(trainresfile);
if isstruct(regModel), %#ok<NODEF>
  regModel = {regModel};
end
ncrossvalsets = numel(regModel);
  
%Test model
phisPr = nan(size(phisTr));
phisPrAll = nan([size(phisTr),prunePrm0.numInit]);

for cvi = 1:ncrossvalsets,
  
  idxtest = cvidx == cvi;
  bboxesTe = repmat(permute(initlocs,[3,1,2]),[nnz(idxtest),1,1]);

  [phisPr(idxtest,:),phisPrAll(idxtest,:,:)] = test_rcpr([],bboxesTe,IsTr(idxtest),regModel{cvi},regPrm,prunePrm0); 
  errcurr = mean( sqrt(sum( (phisPr(idxtest,:)-phisTr(idxtest,:)).^2, 2)) );
  fprintf('Error for validation set %d = %f\n',cvi,errcurr);

end
err = mean( sqrt(sum( (phisPr-phisTr).^2, 2)) );
  

if ~isempty(testresfile),
  
  save(testresfile,'phisPr','phisPrAll','err','trainresfile','paramfile1','paramfile2');
  
end


