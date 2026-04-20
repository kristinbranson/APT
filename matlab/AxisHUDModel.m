
classdef AxisHUDModel < handle
% Model layer for AxisHUD.
%
% Holds non-gobject configuration that AxisHUD (view) reads: which readout
% fields are active, layout constants, colors, and format strings.

  properties
    hasSusp = false  % scalar logical
    hasTrklet = false  % scalar logical

    % % Layout constants
    % txtXoff = 10
    % txtHgt = 17
    % txtWdh = 130
    % annoXoff = 0
    % annoHgt = 34
    % annoWdh = 50
    % 
    % % Color constants
    % txtClrTarget = [1 0.6 0.784]
    % txtClrLblPoint = [1 1 0]
    % txtClrSusp = [1 1 1]
    % txtClrTrklet = [1 1 1]
  end

end  % classdef
