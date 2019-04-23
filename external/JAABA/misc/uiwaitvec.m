function uiwaitvec(h)
while ~isempty(h)
  uiwait(h(1));
  h = h(ishandle(h));
end