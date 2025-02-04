function LabelerGUIControlActuated(source, event)
% General function to handle the actuation of control.
% Simply passes the message on to the controller.
% This means we don't need a closure for every callback.

controlName = source.Tag ;
handles = guidata(source) ;
controller = handles.controller ;
controller.controlActuated(controlName, source, event) ;
