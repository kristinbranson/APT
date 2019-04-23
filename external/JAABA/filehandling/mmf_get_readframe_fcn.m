% readframe = mmf_get_readframe_fcn(header)
%
% Returns the handle to a function which inputs the frame number and
% outputs the corresponding frame for a MMF file described by the input
% 'header'. 
%
% Note that this 'header' is modified by mmf_read_frame and that header
% seems to be stored in the function handle instance somehow. I don't know
% how fragile this is. 
%
function readframe = mmf_get_readframe_fcn(header)

readframe = @(f) mmfreadframe(f);

  function varargout = mmfreadframe(f)
    % this 'header' is stored in the function handle
    if nargout >= 5,
      [im,header,timestamp,bb,mu] = mmf_read_frame(header,f);
    else
      [im,header,timestamp,bb] = mmf_read_frame(header,f);
    end
    cachedidx = header.cachedmeans_idx;
    if nargout >= 5,
      varargout = {im,timestamp,cachedidx,bb,mu};
    else
      varargout = {im,timestamp,cachedidx,bb};
    end
  end

end