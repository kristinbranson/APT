function [P1,pT,lossT] = RCPR_simple_test(testfile,modelfile,outfile)
  
Q = load(testfile);
P = load(modelfile);
nTr = numel(Q.IsTr);
IsTr = Q.IsTr;
if isfield(P,'H0') && ~isempty(P.H0)
  for i=1:nTr,
    IsTr{i}=histeq(IsTr{i},P.H0);
  end
end

[pT,~,lossT]=test_rcpr(Q.phisTr,Q.bboxesTr,IsTr,P.regModel,P.regPrm,P.prunePrm);
P1 = struct;
for ndx = 1:numel(Q.expdirs_all)
  P1.moviefiles_all{ndx} = fullfile(Q.expdirs_all{ndx},'movie_comb.avi');
  P1.p_all{ndx} = pT(Q.phis2dir==ndx,:);
end

P1.minv= 0;
P1.maxv = 255;
save(outfile,'-struct','P1');
