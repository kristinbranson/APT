function errorNoTrace(emsg)
if isdeployed()
  error(emsg);
else
  st = dbstack;
  if numel(st)>=2
    st = st(2);
  else
    st = st(1);
  end
  error(struct('message',emsg,'stack',st));
end