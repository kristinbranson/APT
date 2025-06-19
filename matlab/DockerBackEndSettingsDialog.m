function hfig = DockerBackEndSettingsDialog(obj)

lObj = obj.labeler_;

nrows = 3 + 2;
h = 30;
W = 500;
ysep = 5;
H = (h+ysep)*(nrows+1);

hfig = uifigure('Name','Docker Backend Settings',...
  'Units','pixels','Position',[300 300 W H]);


xborder = 20;
xsep = 5;
yseprel = .025;
wquest = (W-xborder*2-xsep)*.5;
wans = (W-xborder*2-xsep)*.5;
wbutton = 80;
nbuttons = 3;

ycurr = H-ysep-3*h/2;

res = struct;
fns = {'dockerremotehost','dockerimgfull'};
for i = 1:numel(fns),
  res.(fns{i})= lObj.get_backend_property(fns{i});
end
res.isremote = ~isempty(res.dockerremotehost);

hs = struct;

uicontrol('Parent',hfig,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h],...
  'String','Run locally or on a remote host?',...
  'HorizontalAlignment','right');

hs.bg = uibuttongroup(hfig,...
  'Position',[xborder+wquest+xsep,ycurr,wans,h],...
  'BorderType','line',...
  'SelectionChangedFcn',@SetLocalVsRemote);
hs.islocal = uiradiobutton(hs.bg,...
  'Text','Local','Position',[xsep,0,(wans-3*xsep)/2,h]);
hs.isremote = uiradiobutton(hs.bg,...
  'Text','Remote','Position',[wans/2+xsep/2,0,(wans-3*xsep)/2,h]);

ycurr = ycurr - h - ysep;

hs.txt_dockerremotehost = uicontrol('Parent',hfig,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Remote host:',...
  'HorizontalAlignment','right');

hs.dockerremotehost = uicontrol('Parent',hfig,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',res.dockerremotehost,...
  'Callback',@(src,evt) ValidateSetRemoteHost(src,evt)...
  );

if isempty(res.dockerremotehost),
  hs.islocal.Value = true;
  hs.txt_dockerremotehost.Enable = 'off';
  hs.dockerremotehost.Enable = 'off';
else
  hs.isremote.Value = true;
end

ycurr = ycurr - h - ysep;

uicontrol('Parent',hfig,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Docker image:',...
  'HorizontalAlignment','right');

hs.dockerimgfull = uicontrol('Parent',hfig,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',res.dockerimgfull,...
  'Callback',@(src,evt) ValidateSetDockerImg(src,evt)...
  );

ycurr = ycurr - 2*h - ysep;

xcurr = (W-wbutton*nbuttons+xsep*(nbuttons-1))/2;

uicontrol('Parent',hfig,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','Apply',...
  'HorizontalAlignment','center',...
  'Callback',@Apply);

xcurr = xcurr + wbutton + xsep;

uicontrol('Parent',hfig,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','OK',...
  'HorizontalAlignment','center',...
  'Callback',@OK);

xcurr = xcurr + wbutton + xsep;

uicontrol('Parent',hfig,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','Cancel',...
  'HorizontalAlignment','center',...
  'Callback',@Cancel);

centerOnParentFigure(hfig,lObj.gdata.mainFigure_);

  function SetLocalVsRemote(src,evt)
    res.isremote = evt.NewValue == hs.isremote;
    hs.txt_dockerremotehost.Enable = onIff(res.isremote);
    hs.dockerremotehost.Enable = onIff(res.isremote);
  end

  function ValidateSetRemoteHost(src,evt)
    val = strtrim(src.String);
    if isempty(val) && res.isremote,
      warndlg('Remote host cannot be empty','Bad value','modal');
      return;
    end
    res.dockerimgfull = val;
  end

  function ValidateSetDockerImg(src,evt)
    val = strtrim(src.String);
    if isempty(val),
      warndlg('Docker image cannot be empty','Bad value','modal');
      return;
    end
    res.dockerimgfull = val;
  end

  function tfsucc = Apply(varargin)
    tfsucc = true;
    if res.isremote
      val = res.dockerremotehost;
    else
      val = '';
    end
    try
      lObj.set_backend_property('dockerremotehost',val);
    catch ME,
      warndlg(sprintf('Error setting dockerremotehost: %s',getReport(ME)),'Error setting value','modal');
      tfsucc = false;
      res.dockerremotehost = lObj.get_backend_property('dockerremotehost');
      if isempty(res.dockerremotehost),
        hs.islocal.Value = true;
        hs.txt_dockerremotehost.Enable = 'off';
        hs.dockerremotehost.Enable = 'off';
      else
        hs.isremote.Value = true;
      end
    end
        
    fn = 'dockerimgfull';
    try
      lObj.set_backend_property(fn,res.(fn));
    catch ME,
      warndlg(sprintf('Error setting %s: %s',fn,getReport(ME)),'Error setting value','modal');
      res.(fn) = lObj.get_backend_property(fn);
      hs.(fn).String = res.(fn);
      tfsucc = false;
    end

  end

  function OK(varargin)
    tfsucc = Apply(varargin{:});
    if ~tfsucc,
      return;
    end
    Cancel(varargin{:});
  end

  function Cancel(varargin)
    delete(hfig);
  end

end