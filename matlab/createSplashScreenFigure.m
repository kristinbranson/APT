function hfig = createSplashScreenFigure()
% Create the APT splash screen figure

%hparent = handles.figure;
hfig = nan;
splashimfilename = fullfile(APT.Root,'gfx','SplashScreen.png');
if ~exist(splashimfilename,'file'),
  return
end

% oldunits = get(hparent,'Units');
% set(hparent,'Units','pixels');
% pos0 = get(hparent,'Position');
% set(hparent,'Units',oldunits);

warnst = warning('off','MATLAB:imagesci:png:libraryWarning');
im = imread(splashimfilename);
warning(warnst);
sz = size(im);
sz = sz(1:2);

s = {'APT: The Animal Part Tracker'
  'http://kristinbranson.github.io/APT/'
  ''
  'Developed and tested by Allen Lee, Mayank Kabra,'
  'Adam Taylor, Alice Robie, Felipe Rodriguez,'
  'Stephen Huston, Roian Egnor, Austin Edwards,'
  'Caroline Maloney, and Kristin Branson'};

border = 20;
w0 = 400;
texth1 = 25;
w = w0+2*border;
texth2 = (numel(s)-1)*texth1;
textskip = 5;
h0 = w0*sz(1)/sz(2);
h = 2*border+h0+border+texth2+textskip+texth1;

r = [w,h]/2;

center = get(0,'ScreenSize');
center = center(3:4)/2;
%center = pos0([1,2])+pos0([3,4])/2;
pos1 = [center-r,2*r];

hfig = figure('Name','Starting APT...',...
              'Color','k',...
              'Units','pixels',...
              'Position',pos1,...
              'ToolBar','none',...
              'NumberTitle','off',...
              'MenuBar','none',...
              'Pointer','watch');  %'Visible','off',
hax = axes('Parent',hfig,'Units','pixels','Position',[border,border,w0,h0]);
him = image(im,'Parent',hax,'Tag','image_SplashScreen'); axis(hax,'image','off');  %#ok<NASGU> 
htext = ...
  uicontrol('Style','text','String',s{1},'Units','pixels','Position',[border,h-border-texth1,w0,texth1],...
            'BackgroundColor','k','HorizontalAlignment','center',...
            'Parent',hfig,'ForegroundColor','c','FontUnits','pixels','FontSize',texth1*.9,'FontWeight','b',...
            'Tag','text1_SplashScreen'); %#ok<NASGU>
htext = ...
  uicontrol('Style','text','String',s(2:end),'Units','pixels','Position',[border,border+h0+border,w0,texth2],...
            'BackgroundColor','k','HorizontalAlignment','center',...
            'Parent',hfig,'ForegroundColor','c','FontUnits','pixels','FontSize',14,...
            'Tag','text2_SplashScreen'); %#ok<NASGU>
set(hfig,'Visible','on');

end
