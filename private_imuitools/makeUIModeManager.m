function manager = makeUIModeManager(makeDefaultModeCurrent)
%makeUIModeManager Mode manager for managing toolbar items, menu items, modes.
%   MANAGER = makeUIModeManager(makeDefaultModeCurrent) creates a mutually
%   exclusive mode manager and a default mode. MANAGER is a structure of function
%   handles that allows you to work with the mode manager..
%  
%   MANAGER.activateDefaultMode activates the default mode.
%
%   MANAGER.addMode(button,menuItem,makeThisModeCurrent) adds a mode to the mode
%   manager such that the button's 'State' property and the menuItem's 'Checked'
%   property stay in sync and when either is turned on, the corresponding mode
%   is activated by calling the function stored in the function handle
%   makeThisModeCurrent. When either the menu item or the toolbar button is
%   turned off, makeDefaultModeCurrent is called.
%
%   MANAGER.addMode(...,modeChangedCallback) specifies a modeChanged
%   callback that is fired prior to a proposed mode change. modeChangedCallback is a function handle with the signature:
%
%       TF = modeChangedCallback();
%
%   TF is a logical. True means that the proposed mode changed will occur.
%   False means that the proposed mode change will not occur.
%
%   See also makeExclusiveModeManager, 
  
%   Copyright 2005-2008 The MathWorks, Inc.  
%   $Revision: 1.1.6.2 $ $Date: 2008/05/14 21:58:44 $
  
  modeManager = makeExclusiveModeManager;
  
  activateDefaultMode = modeManager.addMode(makeDefaultModeCurrent,...
                                            identityFcn);

  manager.activateDefaultMode = activateDefaultMode;
  manager.addMode             = @addMode;
    
  %----------------------------------------------------------------------------        
  function activateMode = addMode(button,menuItem,makeThisModeCurrent,varargin)
  
    % varargin contains an optional parameter which allows clients to
    % specify a mode changed callback.
    if nargin > 3
        modeChangedCallback = varargin{1};
    else
        modeChangedCallback = @() true;
    end
      
    mode = makeModeFcns(button,menuItem,makeThisModeCurrent);
    activateMode = modeManager.addMode(mode.turnOnMode,...
                                       mode.turnOffMode);
    set(button,'ClickedCallback',@toggleMode)
    set(menuItem,'Callback',@toggleMode)
    
      %----------------------------
      function toggleMode(varargin)

        src = varargin{1};
        srcType = get(src,'Type');
        if strcmp(srcType,'uimenu')
          % If you just clicked on a menu item, it has the 'Checked' 
          % status from previously.
          selectedProperty = 'Checked';
          currentlyOn = strcmp(get(src,selectedProperty),'on');          
          turnOnThisMode = ~currentlyOn;
        else
          % If you just clicked on a toggle button, 
          % it already has the 'State' updated.
          selectedProperty = 'State';
          currentlyOn = strcmp(get(src,selectedProperty),'on');
          turnOnThisMode = currentlyOn;
        end
        
        % Until modeChangedCallback fires, remain in current toolbar state.
        set(src,selectedProperty,'off');
        switchModes = modeChangedCallback();
        
        if ~turnOnThisMode
             % turn off
          activateDefaultMode();
        elseif switchModes
            % turn on
            activateMode();
            set(src,selectedProperty,'on');
        end
           
     end

  end % setUpMode

  %-------------------------------------------------        
  function f = makeModeFcns(button,menuItem,makeThisModeCurrent)
   
    f.turnOnMode  = @turnOnMode;
    f.turnOffMode = @turnOffMode;    

    %-------------------    
    function turnOnMode
    
      toggleOffToOn(button,'State')
      toggleOffToOn(menuItem,'Checked')

      makeThisModeCurrent();
    
    end
  
    %-------------------
    function turnOffMode

      toggleOnToOff(button,'State')
      toggleOnToOff(menuItem,'Checked')      

      makeDefaultModeCurrent()
    
      end
    
  end

end

%--------------------------------
function  toggleOffToOn(src,prop)

  if strcmp(get(src,prop),'off')
    set(src,prop,'on')  
  end

end

%--------------------------------
function  toggleOnToOff(src,prop)

  if strcmp(get(src,prop),'on')
    set(src,prop,'off')  
  end

end