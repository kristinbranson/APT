function hPanel = lockRatioMagPanel(parent,hImLeft,hImRight,leftImageName,rightImageName)
%lockRatioMagPanel Create panel to control magnification of two images.
%   HPANEL = ...
%      lockRatioMagPanel(PARENT,leftImage,rightImage,leftImageName,rightImageName)
%   displays magnification boxes for each image and a lock ratio check box.
%   PARENT is a handle to an object that can contain a uiflowcontainer.
%
%   Arguments returned include:
%      hPanel   - Handle to panel containing two scroll panels
%
%   Example
%   -------
%       % Display two images side-by-side in scroll panels
%       % and set up a magnification panel to control the magnification.
%       left = imread('peppers.png');
%       right = edge(left(:,:,1),'canny');
%       hFig = figure('Toolbar','none',...
%                     'Menubar','none');
%       [hSP,hImL,hImR] = leftRightImscrollpanel(hFig,left,right);
%       hMP = lockRatioMagPanel(hFig,hImL,hImR,'Left Image','Right Image');
%
%       hflow = uiflowcontainer('Parent',hFig,...
%                               'FlowDirection','topdown',...
%                               'Margin',1);
%
%       %Reparent subpanels 
%       set(hMP,'Parent',hflow);
%       set(hMP,'HeightLimits',[30 30]); % pin height
%       set(hSP,'Parent',hflow)

%   Copyright 2005-2008 The MathWorks, Inc.  
%   $Revision: 1.1.6.7 $  $Date: 2011/08/09 17:55:31 $

  hSpLeft = imshared.getimscrollpanel(hImLeft);
  hSpRight = imshared.getimscrollpanel(hImRight);
    
  apiLeft = iptgetapi(hSpLeft);
  apiRight = iptgetapi(hSpRight);
  
  [cbLeft,apiCbLeft] = immagboxjava;
  apiCbLeft.setScrollpanel(apiLeft);
  
  [cbRight,apiCbRight] = immagboxjava;
  apiCbRight.setScrollpanel(apiRight);
  
  % FLOW for titles and mag boxes
  hPanel = uiflowcontainer('v0',...
                           'parent',parent,...
                           'FlowDirection','lefttoright',...
                           'DeleteFcn',@deleteMagPanel);
  
  % Note: Order matters for these 3 calls as they all get parented to the same
  % object, and the order of the calls determines the location in the flow layout.
  hLeftTitlePanel           = uiflowcontainer('v0',...
                                               'Parent',hPanel,...
                                               'FlowDirection','lefttoright');
  hMagBoxesAndCheckBoxPanel = uiflowcontainer('v0',...
                                              'Parent',hPanel,...
                                              'FlowDirection','lefttoright');
  hRightTitlePanel          = uiflowcontainer('v0',...
                                             'Parent',hPanel,...
                                             'FlowDirection','righttoleft');
    
  uicontrol('Parent',hLeftTitlePanel,...
            'Style','text',...
            'HorizontalAlignment','left',...
            'String',leftImageName);
  
  uicontrol('Parent',hRightTitlePanel,...
            'Style','text',...
            'HorizontalAlignment','Right',...
            'String',rightImageName);
     
  % pin size to fit 2 magboxes plus checkbox
  midPanelW = 260;
  set(hMagBoxesAndCheckBoxPanel,'WidthLimits',[midPanelW midPanelW])
  
  % waiting for Bill York's update to javacomponent
  %javacomponentBY(cbLeft, [20 20 100 30], hMagBoxesAndCheckBoxPanel);
  
  hFig = ancestor(parent,'figure');
  
  % Note: Order matters for these calls as they all get parented to the same
  % object, and the order of the calls determines the location in the flow layout.
  
  % Add left mag box to panel
  % Workaround to parent java component to uiflowcontainer
  [dummy, hcLeft] = javacomponent(cbLeft, [20 20 100 30], hFig); %#ok dummy
  set(hcLeft, 'Opaque', 'on', 'Parent', hMagBoxesAndCheckBoxPanel);
  
  % Add spacer panel to leave a little space
  hSpacerPanel = uipanel('BorderType','none',...
                         'Parent',hMagBoxesAndCheckBoxPanel);
  set(hSpacerPanel,'WidthLimits',[2 2]);
   
  % Create "Lock ratio" checkbox and get it looking pretty
  hLockRatioPanel = uipanel('Parent',hMagBoxesAndCheckBoxPanel);
  hLockRatioCheckBox = uicontrol('Parent',hLockRatioPanel,...
                                 'Style','checkbox',...
                                 'String',getString(message('images:privateUIString:lockRatioMagPanelString')),...
                                 'Callback',@updateRatio,...
                                 'Tag','LockRatioCheckBox');
  checkBoxExtent = get(hLockRatioCheckBox,'Extent');
  checkBoxW = checkBoxExtent(3) + 25; % + 25 gives room for box plus text
  set(hLockRatioPanel,'WidthLimits',[checkBoxW checkBoxW]); 
  set(hLockRatioCheckBox,'Position',[0 0 checkBoxW-2 checkBoxExtent(4)])
  
  % Add right mag box to panel
  % Workaround to parent java component to uiflowcontainer
  [dummy, hcRight] = javacomponent(cbRight, [20 20 100 30], hFig); %#ok dummy
  set(hcRight, 'Opaque', 'on', 'Parent', hMagBoxesAndCheckBoxPanel);   

  % Initialize for function scope
  fLeftOverRightMagnification = [];
  setLeftOverRightMagnification

  idMagLeft = apiLeft.addNewMagnificationCallback(@respondToLeftMagChange);
  idMagRight = apiRight.addNewMagnificationCallback(@respondToRightMagChange);  

  %--------------------------------
  function deleteMagPanel(varargin) %#ok varargin needed by HG caller.

    apiLeft.removeNewMagnificationCallback(idMagLeft)
    apiRight.removeNewMagnificationCallback(idMagRight)    
    
  end
  
  %--------------------------------------------
  function respondToLeftMagChange(mag)
    
    if isCheckBoxSelected
      magLeft = mag / getLeftOverRightMagnification;
      apiRight.setMagnification(magLeft)
    end

  end
  
  %--------------------------------------------
  function respondToRightMagChange(mag)
    
    if isCheckBoxSelected
      magRight = mag * getLeftOverRightMagnification;
      apiLeft.setMagnification(magRight)
    end

  end
  
  %-----------------------------
  function updateRatio(varargin) %#ok varargin needed by HG caller.

    if isCheckBoxSelected
      setLeftOverRightMagnification
    end

  end

  %--------------------------------------------
  function setLeftOverRightMagnification

     fLeftOverRightMagnification = apiLeft.getMagnification() / ...
                                   apiRight.getMagnification();

  end

  %--------------------------------------------
  function ratio = getLeftOverRightMagnification

     ratio = fLeftOverRightMagnification;

  end
  
  %--------------------------------------------
  function isSelected = isCheckBoxSelected
    
     isSelected = isequal(get(hLockRatioCheckBox,'Value'),1);
   
  end

end
