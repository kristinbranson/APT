function [tfsucc] = SetTooltip(h,s,jobjs,jobjnames)

isjobj = nargin >= 3;

tfsucc = false;

fns = fieldnames(get(h));
fns = cellfun(@lower,fns,'Uni',0);
if ismember('tooltipstring',fns),
  set(h,'TooltipString',s);
  tfsucc = true;
  return;
end

jh = [];
if isjobj,
  name = get(h,'Tag');
  i = find(strcmp(jobjnames,name));
  if ~isempty(i),
    jh = jobjs(i);
  end
end

try
  if isempty(jh),
    % this is slow, just don't have tooltips when the first pass fails
    return;
    %fprintf('Calling findjobj - %s\n',s);
    %jh = findjobj_modern(h);
  end
  fns = fieldnames(get(jh));
  fns = cellfun(@lower,fns,'Uni',0);
  if ismember('tooltiptext',fns),
    set(jh,'ToolTipText',s);
    tfsucc = true;
  end
end

if ~tfsucc,
  warningNoTrace(sprintf('Could not set tooltip %s',s));
end