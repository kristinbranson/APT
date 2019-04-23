function [readframe,nframes,fid,headerinfo] = get_readframe_fcn_aviread(filename)

headerinfo = aviinfo(filename);
nframes = headerinfo.NumFrames;
fps = headerinfo.FramesPerSecond;

readframe = @(f) flipdim(aviread_helper(filename,f),1);
headerinfo.type = 'avi';
fid = -1;

  function [im,stamp] = aviread_helper(filename,f)
    
    M = aviread(filename,f);
    im = cat(4,M.cdata);
    stamp = f / fps;
    
  end

end