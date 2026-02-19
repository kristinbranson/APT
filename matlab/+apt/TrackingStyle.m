classdef TrackingStyle
  % Enumeration type for the two styles of tracking.
  % movie: Track all frames in one or more movies.
  % list: Track a specific list of (movie, frame, target) tuples.
  enumeration
    movie
    list
  end

  methods
    function result = char(obj)
      result = string(obj).char() ;
    end % function
  end % methods

end % classdef
