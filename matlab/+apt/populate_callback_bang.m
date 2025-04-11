function populate_callback_bang(control, controller)
% Populate the Callback property for a single control, setting to call the
% generic controlActuated(tag, source, event) method on the supplied
% controller.  But only do this if the Callback property is empty---don't
% overwrite existing callbacks.  The "_bang" denotes that this function
% mutates the control.

if isprop(control, 'Callback') && isempty(control.Callback) ,
  tag = control.tag ;
  control.Callback = @(source,event)(controller.controlActuated(tag, source, event)) ;
end

end  % function
