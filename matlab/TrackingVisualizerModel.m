classdef TrackingVisualizerModel < handle
% Abstract base for tracking visualizer model classes.
%
% TrackingVisualizerModel holds the non-gobject (data/model) state that was
% previously stored directly on TrackingVisualizer* view classes.  Concrete
% subclasses correspond to the various TV types (MT, MTFast, Tracklets,
% etc.).
%
% A TVM is owned by DeepTracker (.trkVizer property).  The corresponding TV
% (view) lives on LabelerController and reads model data through
%   obj.parent_.labeler_.tracker.trkVizer
% where parent_ is the LabelerController.

  methods (Abstract)

    trkInit(obj, trk)
    % Initialize tracking data from a TrkFile.

    newFrame(obj, frm)
    % Compute per-frame data for the TV to render.  Returns data that the
    % TV uses to update graphics.

  end  % methods (Abstract)

end  % classdef
