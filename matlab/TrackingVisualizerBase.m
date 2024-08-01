classdef TrackingVisualizerBase < handle
% Tracking Visualizer base class; mostly for doc
%
% TrackingVisualizerBase provides an interface for TVs, currently related
% to frame updates and cosmetics. Clients like Labeler and DeepTracker have 
% common needs regardless of visualization type.
%
% *Frame updates, direct*
% Some clients may desire a direct API for setting predictions. This is
% currently used by eg LabelCoreSeqMA. In this mode:
% - trkInit() and newFrame() are not called
% - eg TrackingVisualizerMT/updateTrackRes is called
%
% TrackingVisualizerBase does not currently have API for this and it will
% be specific to concrete TVs.
%
% *Frame updates, loaded tracking results*
% In this mode, tracking results (ie TrkFile or Trk) are provided
% to TVs via trkInit(). Then frame-updates are handled fully within TVs by
% calling newFrame(). Some points:
%   - new tracking results are generated only infrequently (by either a 
% tracking job, or an import); so trkInit() calls should be fairly sparse
%   - a TV may desire to massage or generate its own custom data structure 
% for viz. For eg trajectory histories, a TV may also maintain internal
% state of frames seen and thus an external controller simply calling set
% at every new frame doesn't feel quite right.
%   - a Trk (TrkFile) + TV by themselves could be operational or nearly so 
% for visualization of results without any controller object.
%
% *Cosmetics*
% TVs are now initialized "on demand" to avoid graphics overhead for
% Labeler nav. After construciton, cosmetics are initialized from Labeler 
% at vizInit()-time. Subsequent modifications to cosmetics occur through 
% the remaining methods.
% 

  methods (Abstract)
    
    vizInit(obj,varargin)
    % initialize graphics handles and cosmetics from Labeler (TVs currently 
    % store a Labeler handle as a prop)
    
    trkInit(obj,trk)
    % initialize tracking results
    % trk: TrkFile
    
    newFrame(obj,frm)
    % display tracking results for given/new frame
    
    initAndUpdateSkeletonEdges(obj,sedges)
    % Inits skel edges and sets their posns based on current hXYPrdRed.

    setShowSkeleton(obj,tf)
  
    setHideViz(obj,tf)
    % If true, hide all viz
    
    setHideTextLbls(obj,tf)
    % Hide only text labels
    
    setAllShowHide(obj,tfHide,tfHideTxt,tfShowCurrTgtOnly)
    
    updateLandmarkColors(obj,ptsClrs)
    
    setMarkerCosmetics(obj,pvargs)
    
    setTextCosmetics(obj,pvargs)
    
    setTextOffset(obj,offsetPx)

    setShowOnlyPrimary(obj,tf)
    
  end
  
end