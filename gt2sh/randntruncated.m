function y = randntruncated(sd,maxabs)

BATCH = 100;
while 1
  y = randn(BATCH,1)*sd;
  idx = find(abs(y)<=maxabs,1);
  if ~isempty(idx)
    y = y(idx);
    return;
  end
end
  