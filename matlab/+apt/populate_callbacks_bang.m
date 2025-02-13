function populate_callbacks_bang(control, controller)
% Populate the Callback property for a single control, setting it to call the
% generic controlActuated(tag, source, event) method on the supplied
% % controller.  But only do this if the Callback property is empty---don't
% overwrite existing callbacks.  Do the same for the Children of control, recursively.  The
% "_bang" denotes that this function mutates the control.

apt.populate_callback_bang(control, controller) ;
if isprop(control, 'Children') ,
  kids = control.Children ;
  arrayfun(@(kid)(apt.populate_callbacks_bang(kid, controller)) , ...
           kids) ;
end

end  % function
