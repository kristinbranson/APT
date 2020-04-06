function convertPPtrkfileToNone(tfilein,tfileout)

t = load(tfilein,'-mat');

szassert(t.pTrk,size(t.pTrkSingleView));
t.pTrk = t.pTrkSingleView;

FLDSRM = {'pTrk3d' 'pTrkSingleView' 'recon3d_prefview'};
for f=FLDSRM,f=f{1};
  if isfield(t,f)
    warningNoTrace('Removing field ''%s''.',f);
    t = rmfield(t,f);
  end
end

save(tfileout,'-mat','-struct','t');
fprintf(1,'Saved %s.\n',tfileout);
