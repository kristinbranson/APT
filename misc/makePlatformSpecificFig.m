function makePlatformSpecificFig(figFile,newFigFile,targetPlatform)
% figFile: name of original .fig file (with extension)
% newFigFile: name of new .fig file (with extension)
% targetPlatform: platform of target.one of {'pc' 'mac' 'unix'}
%
% This file exists to enable cross-platform use of a GUIDE-generated .fig
% with proportional resizing.
%
% Proportional resize requires that we set Units and FontUnits in the .fig
% to be 'normalized'. However, when using a single .fig across platforms,
% cosmetic issues tend to be minimized when settings all units to 'pixels'.
%
% This suggests a workflow where dev is done on a single .fig file (with
% units==pixels) and afterwards, platform-specific, resizeable, fig files
% are generated (by this script.)
%
% This file takes a base, "development" .fig file and generates a
% resizeable, platform-specific, .fig file. The base .fig file should have
% Units='pixels' and FontUnits='pixels' for all uicontrols, uipanels, etc.
% It should be "as good as it gets" for cross-platform usage from a single
% .fig file. This script will convert all Units and FontUnits to
% 'normalized' and resave (to a new name), along with making minor
% platform-specific adjustments as necessary.
%
% Run this script _in the desired target platform_, on a computer with a
% screen large enough to open the BrowserGUI at its 'native' pixel
% resolution.

f = open(figFile);
assert(strcmp(get(f,'Type'),'figure'));

assert(nnz(strcmp(targetPlatform,{'pc' 'mac' 'unix'}))==1);

% Set "outer" figure to have units of pixels, so that all screens that are
% large enough to accomodate the "native" size of the GUI open the GUI with
% that same native size. (If the outer figure had eg units of normalized,
% then its size would be scaled to the screensize.)
set(f,'Units','pixels');

allH = findobj(f);
allH = setdiff(allH,f);

for h = allH(:)'
  
  hType = get(h,'Type');
  
  if strcmp(targetPlatform,'mac')
    % Mac: tweak popup menus
    if strcmp(hType,'uicontrol') && strcmp(get(h,'Style'),'popupmenu')
      pos = get(h,'Position');
      pos(1) = pos(1)-2; % shift PUM to the left by 2 pixels (we assert that original units are ''pixels'', see below)
      set(h,'Position',pos);
    end
  end
  
  % Set units to normalized for all children uicontrols/uipanels etc.
  % This enables proportional rescaling of the GUI.
  if isprop(h,'Units')
    hUnits = get(h,'Units');
    assert(strcmp(hUnits,'pixels'),'''Units'' in original fig file should be ''pixels''.');
    set(h,'Units','Normalized');
  end
  
  % Set FontUnits to normalized
  if isprop(h,'FontUnits')
    hFontUnits = get(h,'FontUnits');
    assert(strcmp(hFontUnits,'pixels'),'''FontUnits'' in original fig should be ''pixels''.');
    
    switch targetPlatform
      case 'pc'
        % AL 20151104 note appears obsolete
        %
        % TMW note: Setting a uipanel's FontUnits to normalized on
        % a PC causes it to hang. Similarly, using a PC to open a
        % .fig file that was created on a Mac, in which a uipanel's
        % FontUnits has been saved as 'Normalized', causes the PC
        % to hang.
%         if ~strcmp(hType,'uipanel')
        set(h,'FontUnits','Normalized');
%         end
      otherwise
        set(h,'FontUnits','Normalized');
    end
  end
end

hgsave(f,newFigFile);

% close opened GUI
delete(f);
