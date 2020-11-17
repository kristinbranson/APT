function about(lObj)

%%
PURPLE = [0.147058823529412 0.033333333333333 0.245098039215686];
WDTH = 900;
HGHT = 420;
h = uifigure(...
  'Position',[1000 1078 WDTH HGHT],...
  'Name','About' ...
  );

if ~isempty(lObj)
  centerOnParentFigure(h,lObj.hFig,'setParentFixUnitsPx',true);
end

ht = uitextarea(h,...
  'BackgroundColor',PURPLE,...
  'HorizontalAlignment','left',... %  'Units','normalized',...
  'Position',[10 10 WDTH-20 HGHT-20] ...
  );

%%
OS = fullfile(APT.Root,'OPENSOURCE');
os = readtxtfile(OS);
for i=numel(os):-1:1
  str = strip(os{i});
  if isempty(str)
    continue;
  end
  if str(1)=='*'
    os(i) = [];
    os{i} = ['*' os{i}(2:end)];
  elseif startsWith(str,'matlab') % special case/hack
    os(i) = [];
  end
end
s = {
  'APT: Animal Part Tracker'
  'Developed in the Branson Lab, Janelia Research Campus'
  ''
  };
s = [s; os];
set(ht,...
  'Value',s,...
  'FontSize',14,...
  'FontColor',[1 1 1]);