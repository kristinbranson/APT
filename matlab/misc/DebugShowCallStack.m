function DebugShowCallStack(src,evt)

stack = dbstack;
for i = 1:length(stack)
    fprintf('%s (line %d)\n', stack(i).name, stack(i).line);
end
if nargin >= 1,
  fprintf('Source:\n');
  disp(src);
end
if nargin >= 2,
  fprintf('Event:\n');
  disp(evt);
end
