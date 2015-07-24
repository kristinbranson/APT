function reactToImageChangesInFig(target_images,h_caller,deleteFcn,refreshFcn)
%reactToImageChangesInFig sets up listeners to react to image changes.
%   reactToImageChangesInFig(TARGET_IMAGES,H_CALLER,DELETE_FCN,REFRESH_FCN)
%   calls DELETE_FCN if any of TARGET_IMAGES are deleted and calls the
%   REFRESH_FCN if the CData property of any of the TARGET_IMAGES is
%   modified.  DELETE_FCN and REFRESH_FCN are function handles specified by
%   the modular tool caller, H_CALLER, to update itself appropriately when
%   its associated image changes.  TARGET_IMAGES is array of handles to
%   graphics image objects.
%
%      DELETE_FCN is called when:
%      ==========================
%      * any of the TARGET_IMAGES are deleted
%
%      REFRESH_FCN is called when:
%      ===========================
%      * the CData property of any of the TARGET_IMAGES is modified
%
%   DELETE_FCN and REFRESH_FCN can optionally be empty, if no action should
%   be taken on these events.
%
%   See also IMPIXELINFO,IMPIXELINFOVAL.

%   Copyright 2004-2010 The MathWorks, Inc.
%   $Revision: 1.1.8.11 $  $Date: 2011/08/09 17:55:33 $

% validate input
narginchk(4,4);
checkImageHandleArray(target_images,mfilename);

% call the deleteFcn if image is destroyed.
if ~isempty(deleteFcn)
    objectDestroyedListener = iptui.iptaddlistener(target_images,...
        'ObjectBeingDestroyed',deleteFcn);
    storeListener(h_caller,'ObjectDestroyedListeners',objectDestroyedListener);
end


% call appropriate function if image cdata changes
if ~isempty(refreshFcn)
    imageCDataChangedListener = iptui.iptaddlistener(target_images,...
        'CData','PostSet',refreshFcn);
    storeListener(h_caller,'CDataChangedListeners',imageCDataChangedListener);
end


%-----------------------------------------------------
function storeListener(h_caller,appdata_name,listener)
% this function stores the listeners in the appdata of the h_caller object
% using the makelist function.

% add to current list of listeners from the caller's appdata
listenerList = getappdata(h_caller,appdata_name);
if isempty(listenerList)
    listenerList = listener;
else
    listenerList(end+1) = listener;
end
setappdata(h_caller,appdata_name,unique(listenerList));

