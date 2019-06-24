function cancelAllFevalQueue(p)
if nargin < 1,
  p = gcp;
end
q = p.FevalQueue;
r = q.QueuedFutures;
for i = 1:numel(r),
  try,
    r(i).cancel();
  catch ME,
    warning(getReport(ME));
  end
end

r = q.RunningFutures;
for i = 1:numel(r),
  try,
    r(i).cancel();
  catch ME,
    warning(getReport(ME));
  end
end