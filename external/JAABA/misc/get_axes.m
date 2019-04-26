function [hax,hfig] = get_axes(hax,hfig,varargin)

[axparams,axi0,axi_create,figi,figpos] = ...
  myparse(varargin,...
  'axparams',{},...
  'axi0',1,'axi_create',[],'figi',1,...
  'figpos',[]);

if isempty(axparams),
  nax = 1;
else
  nax = axparams{1}*axparams{2};
end
if isempty(axi_create),
  axi_create = axi0:axi0+nax-1;
end

if numel(hax) < max(axi_create) || ~all(ishandle(hax(axi_create))),
  % if no ax, check for figure
  if numel(hfig) < figi,
    % if no figure, create
    hfig(figi) = figure;
    if ~isempty(figpos),
      set(hfig(figi),'Position',figpos);
    end
  elseif ishandle(hfig(figi)),
    % there is a figure, so clear it
    clf(hfig(figi));
  else
    % handle input, but figure does not exist, so make it
    figure(hfig(figi));
    if ~isempty(figpos),
      set(hfig(figi),'Position',figpos);
    end
  end
  
  % create the axes
  if isempty(axparams),    
    % single axis
    hax = get(hfig,'CurrentAxes');
    if isempty(hax) || ~ishandle(hax),
      hax(axi_create) = axes('parent',hfig);
    end
  else
    hax(axi_create) = createsubplots(axparams{:},hfig(figi));
  end
  
  % axes exist
else
  hfig(figi) = get(hax(axi_create(1)),'Parent');
end


