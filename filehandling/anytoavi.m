function anytoavi(inmoviename,outmoviename,varargin)

[readframe,nframes,fid] = get_readframe_fcn(inmoviename);
[firstframe,endframe,aviparams] = myparse(varargin,'firstframe',1,'endframe',nframes,...
  'aviparams',{});
UPDATEWAITBARINTERVAL = 20;
firstframe = max(1,min(firstframe,nframes));
endframe = max(1,min(endframe,nframes));

[~,inname,inext] = fileparts(inmoviename);
[~,outname,outext] = fileparts(outmoviename);
inname = [inname,inext];
outname = [outname,outext];
inname = strrep(inname,'_','\_');
outname = strrep(outname,'_','\_');

nframesconvert = endframe-firstframe+1;
h = waitbar(0,sprintf('Converting %s to %s: Frame %d / %d',...
  inname,outname,0,nframesconvert));

im = readframe(firstframe);
im2rgbuint8_fun = get_im2rgbuint8_fun(im,inext);

aviobj = avifile(outmoviename,aviparams{:});

for t = firstframe:endframe,
  if mod(t,UPDATEWAITBARINTERVAL) == 0
    i = t - firstframe + 1;
    if ishandle(h),
      waitbar(i/nframesconvert,h,...
        sprintf('Converting %s to %s: Frame %d / %d',...
        inname,outname,i,nframesconvert));
    end
  end
  im = im2rgbuint8_fun(readframe(t));
  aviobj = addframe(aviobj,uint8(im));
end

aviobj = close(aviobj); %#ok<NASGU>
fclose(fid);
if ishandle(h),
  delete(h);
end

function im2rgbuint8_fun = get_im2rgbuint8_fun(im,inext)

ncolors = size(im,3);
imclass = class(im);
maxvalue = max(im(:));
isfmf = strcmpi(inext,'.fmf') || strcmpi(inext,'.sbfmf') || strcmpi(inext,'.ufmf');

isdouble = maxvalue <= 1 && strcmpi(imclass,'double');
dorep = ncolors == 1;

im2rgbuint8_fun = @(im) im2rgbuint8(im,isdouble,dorep,isfmf);

function imout = im2rgbuint8(imin,isdouble,dorep,isfmf)

if isdouble,
  imout = im2uint8(imin);
else
  imout = uint8(imin);
end

if isfmf,
  imout = flipud(imout);
end

if dorep
  imout = repmat(imout,[1,1,3]);
end
