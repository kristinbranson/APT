function s = replaceInfWithNan(s)      
  % Deal with full-occ rows in s in preparation from generating/writing 
  % TrnPack. infs are written as 'null' to json. match legacy SA
  % behavior by converting infs to nan. 
  
  tfinf = isinf(s.p);
  tfinfX = tfinf(1:s.npts,:);
  tfinfY = tfinf(s.npts+1:end,:);
  assert(isequal(tfinfX,tfinfY),'Label corruption: fully-occluded labels.');
  
%       tf1 = tfinf(1:s.npts,:) | tfinf(s.npts+1:end,:);
%       tf2 = s.occ>0;
%       tfInfWithoutOcc = tf1 & ~tf2;
%       % any point labeled as inf (fully-occ) should have .occ set to true 
%       assert(~any(tfInfWithoutOcc(:),'Label corruption'); 
  
  nfulloccpts = nnz(tfinfX);
  if nfulloccpts>0
    warningNoTrace('Utilizing %d fully-occluded landmarks.',nfulloccpts);
  end
  
  s.p(tfinf) = nan;
  s.occ(tfinfX) = 1;
end  % function
