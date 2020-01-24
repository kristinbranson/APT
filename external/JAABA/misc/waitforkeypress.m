% key = waitforkeypress(keys)
function key = waitforkeypress(keys)

keynums = zeros(size(keys));
for i = 1:length(keys),
  if length(keys{i}) == 1,
    keynums(i) = double(keys{i});
  elseif strcmpi(keys{i},'enter') || strcmpi(keys{i},'return'),
    keynums(i) = 13;
  elseif strcmpi(keys{i},'backspace'),
    keynums(i) = 8;
  elseif strcmpi(keys{i},'left') || strcmpi(keys{i},'leftarrow'),
    keynums(i) = 28;
  elseif strcmpi(keys{i},'right') || strcmpi(keys{i},'rightarrow'),
     keynums(i) = 29;
  elseif strcmpi(keys{i},'up') || strcmpi(keys{i},'uparrow'),
     keynums(i) = 30;
  elseif strcmpi(keys{i},'down') || strcmpi(keys{i},'downarrow'),
     keynums(i) = 31;
  end
end

while true,
  keydown = waitforbuttonpress;
  if ~keydown,
    continue;
  end
  c = get(gcf,'currentcharacter');
  i = find(keynums==double(c),1);
  if ~isempty(i),
    key = keys{i};
    return;
  end
end