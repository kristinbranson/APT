% outfilenames = SaveFigLotsOfWays(hfig,basename,[formats])
function outfilenames = SaveFigLotsOfWays(hfig,basename,formats)

if nargin < 3,
  formats = {'fig','png','pdf','svg'};
end
[path,basename] = myfileparts(basename);
basename0 = basename;

outfilenames = {};

for iteri = 1:2,
  
  if iteri == 2,
    hax = findall(hfig,'type','axes');
    xticklabels = get(hax,'XTickLabel');
    yticklabels = get(hax,'YTickLabel');
    zticklabels = get(hax,'ZTickLabel');
    if numel(hax) == 1,
      xticklabels = {xticklabels};
      yticklabels = {yticklabels};
      zticklabels = {zticklabels};
    end
    for j = 1:numel(hax),
      set(hax(j),'XTickLabel',{},'YTickLabel',{},'ZTickLabel',{});
    end
    htext = setdiff(findall(hfig,'-property','FontSize'),hax);
    isvis = get(htext,'Visible');
    if numel(htext)==1,
      isvis = {isvis};
    elseif isempty(htext),
      isvis = {};
    end
    set(htext,'Visible','off');
    basename = [basename0,'_notext'];
  else
    basename = basename0;
  end
  
  
  if ismember('svg',formats),
    filename = [basename,'.svg'];
    plot2svg(fullfile(path,filename),hfig,'png');
    outfilenames{end+1} = fullfile(path,filename);
  end
  formats = setdiff(formats,{'svg'});
  
  if ismember('ai',formats),
    filename = [basename,'.ai'];
    saveas(hfig,fullfile(path,filename),'ai');
    outfilenames{end+1} = fullfile(path,filename);
  end
  formats = setdiff(formats,{'ai'});
  
  if ismember('fig',formats),
    filename = [basename,'.fig'];
    saveas(hfig,fullfile(path,filename),'fig');
    outfilenames{end+1} = fullfile(path,filename);
  end
  formats = setdiff(formats,{'fig'});
  
  tmpbasename = regexprep(basename,'[^a-zA-Z_0-9]','_');
  for i = 1:numel(formats),
    filename = [basename,'.',formats{i}];
    tmpfilename = [tmpbasename,'.',formats{i}];
    try
      savefig(tmpfilename,hfig,formats{i});
      if ~isempty(path),
        movefile(tmpfilename,fullfile(path,filename));
      end
    catch ME,
      warning('Error while creating %s: %s',filename,getReport(ME));
      continue;
    end
    outfilenames{end+1} = fullfile(path,filename);
  end
  
  if iteri == 2,
    for j = 1:numel(htext),
      set(htext(j),'Visible',isvis{j});
    end
    for j = 1:numel(hax),
      set(hax(j),'XTickLabel',xticklabels{j},'YTickLabel',yticklabels{j},'ZTickLabel',zticklabels{j});
    end

  end
  
end