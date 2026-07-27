classdef CompareTrackersMode
  % The quantity the Compare Trackers window computes and shows bouts of.

  enumeration
    MaximumLandmarkDistance ('Maximum Landmark Distance')
    UnmatchedAnimalCount ('Unmatched Animal Count')
  end

  properties
    prettyStr
  end

  methods
    function obj = CompareTrackersMode(str)
      % Construct a CompareTrackersMode with the given display string.
      obj.prettyStr = str ;
    end  % function
  end  % methods
end  % classdef
