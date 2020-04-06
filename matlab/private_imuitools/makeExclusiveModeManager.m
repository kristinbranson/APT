function s = makeExclusiveModeManager
%makeExclusiveModeManager Make exclusive mode manager.
%   s = makeExclusiveModeManager returns a structure containing function
%   handles for managing a set of related, exclusive modes.  Currently there
%   is only one function handle, stored in the .addMode field.  Its usage is:
%
%       activate_mode_fcn = s.addMode(mode_on_fcn, mode_off_fcn)
%
%   mode_on_fcn and mode_off_fcn are function handles that get called
%   whenever a mode is turned on or turned off, respectively.
%   activate_mode_fcn, the new mode's activation function, is a function
%   handle used to turn the mode on.
%
%   Whenever a mode is activated, the mode manager first calls the previously
%   active mode's mode_off_fcn.  Then the mode manager calls the mode_on_fcn
%   for the mode being activated.
%
%   Example
%   -------
%
%       s = makeExclusiveModeManager;
%       activate_A = s.addMode(@() disp('Mode A on'), @() disp('Mode A off'));
%       activate_B = s.addMode(@() disp('Mode B on'), @() disp('Mode B off'));
%       activate_C = s.addMode(@() disp('Mode C on'), @() disp('Mode C off'));
%       
%       activate_A()
%       activate_B()
%       activate_A()
%       activate_C()
    
%   Copyright 2005 The MathWorks, Inc.  
%   $Revision $  $Date $
    
    s.addMode = @addMode;
    modes = struct('on_fcn', {}, 'off_fcn', {});
    active_mode_index = 0;
    
    function activate_mode_fcn = addMode(mode_on_fcn, mode_off_fcn)

        this_mode_index = numel(modes) + 1;
        modes(this_mode_index).on_fcn = mode_on_fcn;
        modes(this_mode_index).off_fcn = mode_off_fcn;
        
        activate_mode_fcn = @activateMode;
        
        function activateMode(varargin)
            if active_mode_index > 0
                modes(active_mode_index).off_fcn();
            end
            modes(this_mode_index).on_fcn();
            active_mode_index = this_mode_index;
        end
    end
end
    
