function cp = controlPoint(x,y,hAxDetail,hAxOverview,constrainDrag)
%controlPoint 

%   Copyright 2005-2007 The MathWorks, Inc.
%   $Revision: 1.1.6.7 $  $Date: 2007/11/09 20:22:42 $
   
  hDetailPoint   = iptui.cpselectPoint(hAxDetail,x,y,cpPointSymbol,constrainDrag);
  hOverviewPoint = iptui.cpselectPoint(hAxOverview,x,y,cpPointSymbol,constrainDrag);
  
  cp.hDetailPoint     = hDetailPoint;
  cp.hOverviewPoint   = hOverviewPoint;
  cp.setPairId        = @setPairId;
  cp.setActive        = @setActive;
  cp.setPredicted     = @setPredicted;  
  cp.addButtonDownFcn = @addButtonDownFcn;
  
  wireSisterPoint(iptgetapi(hDetailPoint),iptgetapi(hOverviewPoint))
  wireSisterPoint(iptgetapi(hOverviewPoint),iptgetapi(hDetailPoint))   

  %---------------------------------------------------- 
  function wireSisterPoint(movingAPI,sisterAPI)
  % Tell point to move its sister if it moves 
   
   updateSisterPoint = @(pos) sisterAPI.setPosition(pos);
   movingAPI.addNewPositionCallback(updateSisterPoint);
   
  end

  %------------------------------------------------------------------------
  function [detailPointAPI,overviewPointAPI] = getDetailAndOverviewPointAPI
     % This function is provided to avoid the point APIs being declared in the top
     % function workspace. Limiting the workspace of the point APIs yields improved
     % delete performance, especially for larger numbers of control points.
     detailPointAPI = iptgetapi(hDetailPoint);
     overviewPointAPI = iptgetapi(hOverviewPoint);
      
  end    
 
  %---------------------
  function setPairId(id)
      
    [detailPointAPI,overviewPointAPI] = getDetailAndOverviewPointAPI();  
    
    detailPointAPI.setPairId(id)
    overviewPointAPI.setPairId(id)   

  end

  %---------------------------
  function setActive(isActive)

    [detailPointAPI,overviewPointAPI] = getDetailAndOverviewPointAPI();  
      
    % approach to use for "active as adjective" of point    
    detailPointAPI.setActive(isActive)
    overviewPointAPI.setActive(isActive)   
    
  end

  %---------------------------------
  function setPredicted(isPredicted)
      
    [detailPointAPI,overviewPointAPI] = getDetailAndOverviewPointAPI();  

    detailPointAPI.setPredicted(isPredicted)
    overviewPointAPI.setPredicted(isPredicted)  
    
  end
  
  %-----------------------------
  function addButtonDownFcn(fun)

    hDetailPoint.addButtonDownFcn(fun);
    hOverviewPoint.addButtonDownFcn(fun);

  end
  
end
