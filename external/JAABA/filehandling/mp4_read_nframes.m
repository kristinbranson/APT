function nframes = mp4_read_nframes(filename)
% Return the exact frame count of an ISO base-media file (.mp4/.mov/.m4v) by
% traversing its container metadata, without decoding any frames.  This is a
% thin wrapper around mp4_read_index; see that function for details.
%
% Errors if the file cannot be opened, cannot be parsed as ISO base-media, or
% has no video track with a frame count.  Callers that want a graceful
% fallback (e.g. to VideoReader) should wrap the call in try/catch.

  index = mp4_read_index(filename) ;
  nframes = index.nframes ;
end  % function
