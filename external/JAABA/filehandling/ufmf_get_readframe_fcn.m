% readframe = ufmf_get_readframe_fcn(header)
%
% Returns the handle to a function which inputs the frame number and
% outputs the corresponding frame for a UFMF file described by the input
% 'header'. 
%
% Note that this 'header' is modified by ufmf_read_frame and that header
% seems to be stored in the function handle instance somehow. I don't know
% how fragile this is. 
%
function readframe = ufmf_get_readframe_fcn(header,varargin)

[interruptible] = myparse(varargin,'interruptible',false);

readframe = @(f) ufmfreadframe(f);

  function varargout = ufmfreadframe(f)
    % this 'header' is stored in the function handle
    if nargout >= 5,
      [im,header,timestamp,bb,mu] = ufmf_read_frame(header,f,interruptible);
    else
      [im,header,timestamp,bb] = ufmf_read_frame(header,f,interruptible);
    end
    cachedidx = header.cachedmeans_idx;
    if nargout >= 5,
      varargout = {im,timestamp,cachedidx,bb,mu};
    else
      varargout = {im,timestamp,cachedidx,bb};
    end
  end

end