function cpAPI = cpManager(cpstruct,hImInput,hImOvInput,hImBase,hImOvBase,...
                           editMenuItems,pointItems)
%cpManager Manage state of all control points.

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.8 $  $Date: 2008/06/16 16:40:07 $

% Initialize needToSave boolean
needToSave = false;

% Initialize activePairId
activePairId = 0;

constrainInputPoint = makeDragConstraintFcn(hImInput);
constrainBasePoint = makeDragConstraintFcn(hImBase);

% Function scope variable used to manage arrow key adjustment of control
% points.
sideLastSelected = '';

hAxInput   = iptancestor(hImInput,'axes');
hAxOvInput = iptancestor(hImOvInput,'axes');
hAxBase    = iptancestor(hImBase,'axes');
hAxOvBase  = iptancestor(hImOvBase,'axes');

hFig       = iptancestor(hImOvBase,'figure');

% initialize empty pairs struct
inputBasePairs = struct('id',{},...
                        'inputPoint',{},...
                        'basePoint',{},...
                        'predictedPoint',{});

cpFactory = makeControlPointFactory(@changeActivePair);

if isempty(cpstruct)
  initialId = 1;
  
else  
  % Preallocate
  nPairs = length(cpstruct.ids);
  inputBasePairs(nPairs).id = [];
  
  % Check to see that all cpstruct.ids are positive. If a cpstruct was saved in
  % Java-based cpselect, there could be an id with value 0. If we find one like
  % that, add one to all ids.
  idZero = find(cpstruct.ids==0,1);
  if ~isempty(idZero)
    cpstruct.ids = cpstruct.ids + 1;
  end
  
  for i = 1:nPairs
    inputBasePairs(i).id = cpstruct.ids(i);
  end

  if ~isempty(cpstruct.inputPoints)
    inputPoints = createPoints(cpstruct.inputPoints,...
                                 hAxInput,hAxOvInput,constrainInputPoint);
    fillPairs(inputPoints,'inputIdPairs','inputPoint','isInputPredicted');
  end  
  
  if ~isempty(cpstruct.basePoints)
    basePoints  = createPoints(cpstruct.basePoints,...
                                 hAxBase,hAxOvBase,constrainBasePoint);
    fillPairs(basePoints,'baseIdPairs','basePoint','isBasePredicted');
  end
  
  initialId = max(cpstruct.ids) + 1;

end % else

pairIdStream = idStream(initialId);

cpAPI.getInputBasePairs        = @getInputBasePairs;
cpAPI.addInputPoint            = @addInputPoint;
cpAPI.addBasePoint             = @addBasePoint;
cpAPI.addInputPointPredictBase = @addInputPointPredictBase;
cpAPI.addBasePointPredictInput = @addBasePointPredictInput;
cpAPI.getNeedToSave            = @getNeedToSave;
cpAPI.getKeyPressId            = @getKeyPressId;
cpAPI.getWindowKeyPressId      = @getWindowKeyPressId;

% Set up delete point callbacks
set(editMenuItems.deleteActivePair,      'Callback',@deleteActivePair)
set(editMenuItems.deleteActiveInputPoint,'Callback',@deleteActiveInputPoint)
set(editMenuItems.deleteActiveBasePoint, 'Callback',@deleteActiveBasePoint)

% Wire function to manage key press events
keyPressCallbackId = iptaddcallback(hFig,'KeyPressFcn',...
    @manageKeyPress);
windowKeyPressCallbackId = iptaddcallback(hFig,'WindowKeyPressFcn',...
    @manageWindowKeyPress);

% initialize state of tool
pairsChanged

  %--------------------------------------------------------------------
  function fillPairs(points,pointIdPairsName,pointName,isPredictedName)
  % Use dyanamic field names to fill in appropriate fields of inputBasePairs

    pointIdPairs = cpstruct.(pointIdPairsName);
    
    % loop over pairs to set appropriate point for pairs
    nPairs = size(pointIdPairs,1);
    for i = 1:nPairs
      
      [pointIndex, pairIndex] = getPointPairIndex(...
        pointIdPairs(i,:),...
        cpstruct.ids,inputBasePairs);
        
      point = points(pointIndex);
      id = inputBasePairs(pairIndex).id;
      point.setPairId(id)

      inputBasePairs(pairIndex).(pointName) = points(pointIndex);
      if cpstruct.(isPredictedName)(pointIndex)
        inputBasePairs(pairIndex).predictedPoint = points(pointIndex);
        point.setPredicted(true)
      end      
      
    end
  
  end % fillPairs

  %---------------------------------
  function pairs = getInputBasePairs

    pairs = inputBasePairs;
    
  end

  %----------------------------
  function need = getNeedToSave

    need = needToSave;
    
  end

  %----------------
  function id = getKeyPressId
    
    id = keyPressCallbackId;
        
  end

  %----------------
  function id = getWindowKeyPressId
    
    id = windowKeyPressCallbackId;
        
  end

  %------------------------------------
  function [pair,index] = getActivePair

    pair = [];
    
    if (activePairId > 0)
      [pair,index] = findPair(inputBasePairs,activePairId);  
    end
    
  end
  
  %-----------------------------------
  function addInputPoint(obj,varargin)

    hAx = iptancestor(obj,'axes');    
    [x,y] = getCurrentPoint(hAx);

    pairId = whatToDoWithNewInputPoint;    
    cp = createInputPoint(x,y,pairId);
        
    changeActivePair([],[],cp.hDetailPoint);
    
    side = 'input';
    setSideLastSelected(side);
    addLastSelectedSideCallback(cp,side);
    
  end

  %----------------------------------
  function addBasePoint(obj,varargin)

    hAx = iptancestor(obj,'axes');    
    [x,y] = getCurrentPoint(hAx);

    pairId = whatToDoWithNewBasePoint;            
    cp = createBasePoint(x,y,pairId);    
    
    changeActivePair([],[],cp.hDetailPoint)
    
    side = 'base';
    setSideLastSelected(side);
    addLastSelectedSideCallback(cp,side);
    
  end

  %----------------------------------------------
  function addInputPointPredictBase(obj,varargin)

    hAx = iptancestor(obj,'axes');    
    [x,y] = getCurrentPoint(hAx);

    [pairId,addingNewPair] = whatToDoWithNewInputPoint;    
    cp = createInputPoint(x,y,pairId);    
    
    side = 'input';
    setSideLastSelected(side);
    addLastSelectedSideCallback(cp,side);

    if addingNewPair
      predictBase = true; % predict base!
      createPredictedPoint(x,y,pairId,predictBase,@createBasePoint)      
    end
           
    changeActivePair([],[],cp.hDetailPoint)
    
  end

  %----------------------------------------------
  function addBasePointPredictInput(obj,varargin)

    hAx = iptancestor(obj,'axes');    
    [x,y] = getCurrentPoint(hAx);

    [pairId,addingNewPair] = whatToDoWithNewBasePoint;        
    cp = createBasePoint(x,y,pairId);
 
    side = 'base';
    setSideLastSelected(side)
    addLastSelectedSideCallback(cp,side)

    if addingNewPair
      predictBase = false; % predict input!
      createPredictedPoint(x,y,pairId,predictBase,@createInputPoint)      
    end
    
    changeActivePair([],[],cp.hDetailPoint)
        
  end

  %--------------------------------------------
  function addLastSelectedSideCallback(cp,side)
    
    cp.addButtonDownFcn(@updateSideSelected)

    %------------------------------------
    function updateSideSelected(varargin)
        
        setSideLastSelected(side);
        
    end
      
  end

  %---------------------------------
  function setSideLastSelected(side)
     
      sideLastSelected = side;
      
  end    
  
  %-----------------------------------------
  function cp = createInputPoint(x,y,pairId)
    
    cp = cpFactory.new(x,y,hAxInput,hAxOvInput,constrainInputPoint);

    pairIndex = findPairIndex(inputBasePairs,pairId);      
    cp.setPairId(pairId)

    inputBasePairs(pairIndex).inputPoint = cp; 

  end

  %----------------------------------------
  function cp = createBasePoint(x,y,pairId)
    
    cp = cpFactory.new(x,y,hAxBase,hAxOvBase,constrainBasePoint);

    pairIndex = findPairIndex(inputBasePairs,pairId);      
    cp.setPairId(pairId)

    inputBasePairs(pairIndex).basePoint = cp;
    
  end

  %--------------------------------------------------------------
  function createPredictedPoint(x,y,pairId,predictBase,createFcn)
    
    % only predict when adding new pair
    cpstructCurrent = cppairsvector2cpstruct(inputBasePairs);
    [inputPoints,basePoints] = cpstruct2pairs(cpstructCurrent);

    xyPred = cppredict([x y],inputPoints,basePoints,predictBase,...
                       constrainInputPoint,constrainBasePoint);
    
    cpPredicted = createFcn(xyPred(1),xyPred(2),pairId);      

    cpPredicted.setPredicted(true)
	
    % Set up callback that will keep track of which side (input/base) was
    % clicked on when a point is made active. This information is needed
    % for arrow key movement of control points.
    if predictBase
        addLastSelectedSideCallback(cpPredicted,'base')
    else
        addLastSelectedSideCallback(cpPredicted,'input')
    end
      
    pairIndex = findPairIndex(inputBasePairs,pairId);                
    inputBasePairs(pairIndex).predictedPoint = cpPredicted;      
    wirePredicted(cpPredicted)
    
  end

  %----------------------------------
  function wirePredicted(cpPredicted)
      
    detailPointAPI = iptgetapi(cpPredicted.hDetailPoint);

    idDetail = detailPointAPI.addNewPositionCallback(@acceptPredicted);

    insideAcceptPredicted = false;
    
    %----------------------------
    function acceptPredicted(pos) %#ok caller expects single input
      
      % Pattern to break recursion
      if insideAcceptPredicted
          return
      else
          insideAcceptPredicted = true;
      end     

      pairId = detailPointAPI.getPairId();
      pairIndex = findPairIndex(inputBasePairs,pairId);  
      
      inputBasePairs(pairIndex).predictedPoint = [];

      cpPredicted.setPredicted(false);
      
      detailPointAPI.removeNewPositionCallback(idDetail);
    
      % Pattern to break recursion
      insideAcceptPredicted = false;
      
    end
    
  end
  
  %-----------------------------------------------------------
  function [pairId,addingNewPair] = whatToDoWithNewInputPoint
    
    activePair = getActivePair();

    joinActivePair = ~isempty(activePair) && ~isempty(activePair.basePoint) && ...
          isempty(activePair.inputPoint);

    pairId = getPairId(joinActivePair);    
    
    % If we can't join the active pair, we have to make a new one
    addingNewPair = ~joinActivePair;

  end
  
  %----------------------------------------------------------
  function [pairId,addingNewPair] = whatToDoWithNewBasePoint
    
    activePair = getActivePair();

    joinActivePair =  ~isempty(activePair) && ~isempty(activePair.inputPoint) && ...
          isempty(activePair.basePoint);

    pairId = getPairId(joinActivePair);    

    % If we can't join the active pair, we have to make a new one
    addingNewPair = ~joinActivePair;
    
  end

  %------------------------------------------
  function pairId = getPairId(joinActivePair)
    
    activePair = getActivePair();
    
    if joinActivePair
      pairId = activePair.id;
    else
      % make a new pair      
      pairId = makeNewPair();
    end

  end
    
  %-------------------------------
  function newPairId = makeNewPair
  
    % Make new pair, add it to inputBasePairs extending the strucutre array by
    % one. Note, the index may be different from the id returned by
    % pairIdStream.
      
    newPairIndex = length(inputBasePairs) + 1;
    newPairId = pairIdStream.nextId();
    inputBasePairs(newPairIndex).id = newPairId;

  end
 
  %-----------------------------------------------
  function changeActivePair(h_obj,ed,clickedPoint) %#ok varagin needed by HG caller
    
    pairId = clickedPoint.getPairId();
    
    activePair = findPair(inputBasePairs,pairId);  
      
    clearLastActivePair();
    
    setActivePair(activePair,true)

    activePairId = activePair.id;
    pairsChanged();
          
  end
  
  %---------------------------
  function clearLastActivePair
    
    if activePairId > 0

      oldActivePair = findPair(inputBasePairs,activePairId);        

      setActivePair(oldActivePair,false)

    end
  end

  %--------------------
  function pairsChanged

    enableDeleteActivePair = logical2onoff(activePairHasBothPoints());
    enableDeleteActiveInputPoint = logical2onoff(activePairHasInputPoint());
    enableDeleteActiveBasePoint = logical2onoff(activePairHasBasePoint());

    set(editMenuItems.deleteActivePair,...
        'Enable',enableDeleteActivePair)
    set(editMenuItems.deleteActiveInputPoint,...
        'Enable',enableDeleteActiveInputPoint)
    set(editMenuItems.deleteActiveBasePoint,...
        'Enable',enableDeleteActiveBasePoint)    
    
    % count valid pairs
    nValidPairs = countValidPairs(inputBasePairs);

    % Update predict menu and toolbar depending on number of valid pairs
    enoughPairsToPredict = nValidPairs >= 2;

    enablePrediction = logical2onoff(enoughPairsToPredict);
    set([pointItems.addPredictMenuItem pointItems.addPredictButton],...
        'Enable',enablePrediction)
    
    isPredictModeOn = strcmp(get(pointItems.addPredictButton,'State'),'on');
    if ~enoughPairsToPredict && isPredictModeOn
      pointItems.activateAddPointMode()
    end
    
    % Update needToSave based on whether there are any valid pairs
     if nValidPairs > 0
         needToSave = true;
     else
         needToSave = false;
     end    
    
end

  %---------------------------------------
  function isPairEmpty = isActivePairEmpty
      
      activePair = getActivePair();
      isPairEmpty = isempty(activePair);
    
  end
  
  %----------------------------------------------
  function pairHasInput = activePairHasInputPoint
      
      activePair = getActivePair();
      pairHasInput = ~isActivePairEmpty() && ~isempty(activePair.inputPoint);
      
  end

  %--------------------------------------------
  function pairHasBase = activePairHasBasePoint
      
      activePair = getActivePair();
      pairHasBase = ~isActivePairEmpty() && ~isempty(activePair.basePoint);
      
  end
  
  %---------------------------------------------
  function pairHasBoth = activePairHasBothPoints
      
      pairHasBoth = activePairHasInputPoint() && activePairHasBasePoint();
      
  end
      
  %-------------------------
  function recycleActivePair
  
    [activePair, activePairIndex] = getActivePair();

    activePairId = 0;
    pairIdStream.recycleId(activePair.id);

    inputBasePairs(activePairIndex) = []; 

  end
    
    %------------------------------
    function manageKeyPress(src,ed) %#ok<INUSL>
        % this KeyPressFcn handles the arrow keys.  These need to be on the
        % figure's KeyPressFcn to avoid focus issues with the scrollbars.

        switch (ed.Key)
          case {'leftarrow','uparrow','downarrow','rightarrow'}
            arrowKeyPress(ed.Key,ed.Modifier);
          otherwise
            % Do nothing. It is not a key we handle currently.
        end
    end
          
    %------------------------------------
    function manageWindowKeyPress(src,ed) %#ok<INUSL>
        % this is the WindowKeyPressFcn.  We handle delete and backspace
        % here so they remain active regardless of uicontrol focus.

        switch (ed.Key)
          case {'delete','backspace'}
            deleteKeyPress();
          otherwise
            % Do nothing. It is not a key we handle currently.
        end
    end
          
  %-----------------------------------
  function arrowKeyPress(key,modifier)
      
      if strcmp(modifier,'shift')
          delta = 10;
      else
          delta = 1;
      end
      
      switch (key)
        case 'leftarrow'
          deltaPos = [-delta 0];
        case 'rightarrow'
          deltaPos = [delta 0];
        case 'uparrow'
          deltaPos = [0 -delta];
        case 'downarrow'
          deltaPos = [0 delta];  
        otherwise
          % Do nothing, you can't reach this point in the code.
      end
      
	  activePair = getActivePair();
	  
      if (activePairHasBothPoints())
          lastSelectedInputPoint = strcmp(sideLastSelected,'input');
          if lastSelectedInputPoint
              changePointPosition(activePair.inputPoint,deltaPos,constrainInputPoint);
          else
              changePointPosition(activePair.basePoint,deltaPos,constrainBasePoint);
          end    
      elseif (activePairHasInputPoint())
         changePointPosition(activePair.inputPoint,deltaPos,constrainInputPoint);
      elseif (activePairHasBasePoint())
         changePointPosition(activePair.basePoint,deltaPos,constrainBasePoint);
      end
      
     %------------------------------------------------------------------
     function changePointPosition(hCpPoint,deltaPos,dragConstraintFcn)
     % Only want to move detail or overviewpoint associated with a control point.
     % Movement of a detail or overview point causes the associated sister
     % point to move.
         detailPointAPI = iptgetapi(hCpPoint.hDetailPoint);
         detailPointAPI.setPosition(dragConstraintFcn(detailPointAPI.getPosition() + deltaPos));
     end  %end changePointPosition
     
  end %end deleteKeyPress
    
    
  %----------------------
  function deleteKeyPress
      
    % If delete key pressed, delete the active pair if there is one.  Otherwise,
    % there can be only an active input point or an active base point.  
    % If either exist, delete the active base or input point.
    if (activePairHasBothPoints())
        deleteActivePair();
    elseif (activePairHasInputPoint())
        deleteActiveInputPoint();
    elseif (activePairHasBasePoint())
        deleteActiveBasePoint();
    end  
      
  end
       
  
  %----------------------------------
  function deleteActivePair(varargin)

    [activePair,activePairIndex] = getActivePair();

    if ~isempty(activePair)
      inputPoint = activePair.inputPoint;
      delete([inputPoint.hDetailPoint inputPoint.hOverviewPoint])
      inputBasePairs(activePairIndex).inputPoint = [];

      basePoint = activePair.basePoint;
      delete([basePoint.hDetailPoint basePoint.hOverviewPoint])
      inputBasePairs(activePairIndex).basePoint = [];
      
      recycleActivePair
    end          
            
    pairsChanged();
    
  end

  %----------------------------------------
  function deleteActiveInputPoint(varargin)

    [activePair,activePairIndex] = getActivePair();

    if ~isempty(activePair)
      inputPoint = activePair.inputPoint;
      delete([inputPoint.hDetailPoint inputPoint.hOverviewPoint])
      inputBasePairs(activePairIndex).inputPoint = [];

      if isempty(activePair.basePoint)
        recycleActivePair
      end
      
    end          
            
    pairsChanged();
    
  end

  %---------------------------------------
  function deleteActiveBasePoint(varargin)

    [activePair,activePairIndex] = getActivePair();

    if ~isempty(activePair)
      basePoint = activePair.basePoint;
      delete([basePoint.hDetailPoint basePoint.hOverviewPoint])
      inputBasePairs(activePairIndex).basePoint = [];

      if isempty(activePair.inputPoint)
        recycleActivePair
      end
      
    end          
            
    pairsChanged();
    
  end
  
  %-------------------------------------------------------------------------
  function points = createPoints(coords,hAxDetail,hAxOverview,constrainDrag)
  
    points = makeEmptyPointsStruct;
      
    % preallocate
    nPoints = size(coords,1);
    points(nPoints).hDetailPoint = [];
      
    for i = 1:nPoints
      
      xy = coords(i,:);
      x = xy(1);
      y = xy(2);
      
      points(i) = cpFactory.new(x,y,hAxDetail,hAxOverview,constrainDrag);
        
    end
        
  end % createPoints
    
end % cpManager

%------------------------------------------------------
function [pointIndex, pairIndex] = ...
      getPointPairIndex(pointIdPair,ids,inputBasePairs)

  pointIndex = pointIdPair(1);  
  idIndex = pointIdPair(2);
  id = ids(idIndex);
    
  pairIndex = findPairIndex(inputBasePairs,id);  

end

%----------------------------------------------------------
function factory = makeControlPointFactory(changeActiveFun)
% makeControlPointFactory allows certain things to be set for every new
% controlPoint, like the changeActiveFun is a callback now when any point is
% clicked on.
  
  factory.new = @new;

  %--------------------------------------------------------
  function p = new(x,y,hAxDetail,hAxOverview,constrainDrag)
    p = controlPoint(x,y,hAxDetail,hAxOverview,constrainDrag);
    p.addButtonDownFcn(changeActiveFun)
    
  end

end

%---------------------------------
function s = makeEmptyPointsStruct

  s = struct('hDetailPoint',{},...
             'hOverviewPoint',{},...
             'setPairId',{},...
             'setActive',{},...
             'setPredicted',{},...             
             'addButtonDownFcn',{});

end

%------------------------------------
function setActivePair(pair,isActive)
  
  if ~isempty(pair.inputPoint)
    pair.inputPoint.setActive(isActive)
  end
  
  if ~isempty(pair.basePoint)
    pair.basePoint.setActive(isActive)      
  end

end

%-----------------------------------------------------
function nValidPairs = countValidPairs(inputBasePairs)

  nValidPairs = 0;
  for i = 1:length(inputBasePairs)
    pair = inputBasePairs(i);
    if ~isempty(pair.inputPoint) && ~isempty(pair.basePoint) && ... 
          isempty(pair.predictedPoint)
      nValidPairs = nValidPairs + 1;
    end
  end
  
end

%----------------------------------------------------
function pairIndex = findPairIndex(inputBasePairs,id)
% find the index of the pair that has id as its id
  
  pairIndex = find([inputBasePairs(:).id] == id);

end

%------------------------------------------------------
function [pair,pairIndex] = findPair(inputBasePairs,id)
% find the pair and index of the pair that has id as its id
  
  pairIndex = find([inputBasePairs(:).id] == id);
  pair = inputBasePairs(pairIndex);
    
end

%--------------------------------------------------
function constrainDrag = makeDragConstraintFcn(hIm)
  
  % get image dimensions
  imModel = getimagemodel(hIm);
  imageHeight = getImageHeight(imModel);
  imageWidth  = getImageWidth(imModel);
    
  % Create drag constraint function to keep points in image
  constrainDrag = @(pos) constrainPoint(pos,imageWidth,imageHeight);

end
