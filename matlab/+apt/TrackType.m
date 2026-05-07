classdef TrackType
  % Enumeration type for the three types of batch tracking.
  % track: Full tracking (detect + link).
  % id_link: ID-based link only.
  % detect: Detect only.
  enumeration
    track
    id_link
    detect
  end

  methods
    function result = char(obj)
      result = string(obj).char() ;
    end % function
  end % methods

end % classdef
