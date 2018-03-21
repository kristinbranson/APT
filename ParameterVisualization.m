classdef ParameterVisualization < handle
  % Parameters that want a visualization when they are being set should
  % subclass this abstract base class
  
  methods (Abstract)
    
    % Called when a property is selected for consideration, eg when a user
    % clicks a row in the parameter tree
    %
    % hAx: scalar axes. Concrete ParameterVisualizations draw here as
    % desired.
    % imLbl: [nrxnc]. full frame/image for view 1, for some
    % reasonably-selected labeled frame (eg the current frame if possible).
    % pLbl: [nphysptsx2]. labeled points for view 1, same frame as imLbl.
    % movLbl,frmLbl,tgtLbl. movie/frame/target for imLbl/pLbl.
    % lObj
    % sPrm0: starting/current params in UI
    init(hAx,imL,pL,movL,frmL,iTgtL,lObj,sPrm0)
    
    % Called when a user sets/selects a new property value
    %
    % All args are as in init(...), except sPrm contains the latest/updated
    % parameters.
    update(hAx,imL,pL,movL,frmL,iTgtL,lObj,sPrm)
    
  end
  
end