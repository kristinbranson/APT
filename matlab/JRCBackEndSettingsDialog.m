function hdlg = JRCBackEndSettingsDialog(obj)

lObj = obj.labeler_;

nrows = 5 + 2;
h = 30;
W = 500;
H = h*(nrows+1);

hdlg = dialog('Name','JRC Backend Settings',...
  'Units','pixels','Position',[300 300 W H]);


xborder = 20;
xsep = 5;
yseprel = .025;
wquest = (W-xborder*2-xsep)*.5;
wans = (W-xborder*2-xsep)*.5;
wbutton = 80;
nbuttons = 3;


ycurr = H-3*h/2;

res = struct;
fns = {'jrcnslots','jrcnslotstrack','jrcAdditionalBsubArgs','jrcgpuqueue','singularity_image_path'};
for i = 1:numel(fns),
  res.(fns{i})= lObj.get_backend_property(fns{i});
end

hs = struct;

uicontrol('Parent',hdlg,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Training CPU cores:',...
  'HorizontalAlignment','right');

hs.jrcnslots = uicontrol('Parent',hdlg,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',num2str(res.jrcnslots),...
  'Callback',@(src,evt) ValidateSetNSlots(src,evt,'jrcnslots')...
  );

ycurr = ycurr - h;

uicontrol('Parent',hdlg,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Tracking CPU cores:',...
  'HorizontalAlignment','right');

hs.jrcnslotstrack = uicontrol('Parent',hdlg,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',num2str(res.jrcnslotstrack),...
  'Callback',@(src,evt) ValidateSetNSlots(src,evt,'jrcnslotstrack')...
  );

ycurr = ycurr - h;

uicontrol('Parent',hdlg,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Additional bsub arguments:',...
  'HorizontalAlignment','right');

hs.jrcAdditionalBsubArgs = uicontrol('Parent',hdlg,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',res.jrcAdditionalBsubArgs,...
  'Callback',@(src,evt) ValidateSetBsubArgs(src,evt)...
  );


ycurr = ycurr - h;

uicontrol('Parent',hdlg,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','GPU queue:',...
  'HorizontalAlignment','right');

hs.jrcgpuqueue = uicontrol('Parent',hdlg,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',res.jrcgpuqueue,...
  'Callback',@(src,evt) ValidateSetGPUQueue(src,evt)...
  );


ycurr = ycurr - h;

uicontrol('Parent',hdlg,...
  'Style','text',...
  'Position',[xborder,ycurr,wquest,h*(1-yseprel)],...
  'String','Singularity image:',...
  'HorizontalAlignment','right');

hs.singularity_image_path = uicontrol('Parent',hdlg,...
  'Style','edit',...
  'Position',[xborder+wquest+xsep,ycurr,wans,h*(1-yseprel)],...
  'String',res.singularity_image_path,...
  'ButtonDownFcn',@(src,evt) ChooseFile(src,evt,'singularity_image_path'),...
  'Callback',@(src,evt) ValidateSetFile(src,evt,'singularity_image_path'),...
  'Enable','off'...
  );

ycurr = ycurr - 2*h;

xcurr = (W-wbutton*nbuttons+xsep*(nbuttons-1))/2;

uicontrol('Parent',hdlg,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','Apply',...
  'HorizontalAlignment','center',...
  'Callback',@Apply);

xcurr = xcurr + wbutton + xsep;

uicontrol('Parent',hdlg,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','OK',...
  'HorizontalAlignment','center',...
  'Callback',@OK);

xcurr = xcurr + wbutton + xsep;

uicontrol('Parent',hdlg,...
  'Style','pushbutton',...
  'Position',[xcurr,ycurr,wbutton,h*(1-yseprel)],...
  'String','Cancel',...
  'HorizontalAlignment','center',...
  'Callback',@Cancel);



centerOnParentFigure(hdlg,lObj.gdata.mainFigure_);

  function ValidateSetNSlots(src,evt,fn)

    val = str2double(src.String);
    if isnan(val) || val<=0
      src.String = num2str(res.(fn));
      warndlg('Number of CPU cores must be an integer greater than 0','Bad value','modal');
    else
      res.(fn) = val;
    end

  end

  function ValidateSetBsubArgs(src,evt)

    res.jrcAdditionalBsubArgs = src.String;

  end

  function ValidateSetGPUQueue(src,evt)
    new_value = src.String;
    if ~ischar(new_value) || isempty(new_value) || ~startsWith(new_value,'gpu_'),
      warndlg('GPU queue must be a string that starts with "gpu_"','Bad value','modal');
      return;
    end
    res.jrcgpuqueue = new_value;
  end

  function ChooseFile(src,evt,fn)
    original_value = src.String;
    filter_spec = {'*.sif','Singularity Images (*.sif)'; ...
      '*',  'All Files (*)'} ;
    [file_name, path_name] = uigetfile(filter_spec, 'Set Singularity Image...', original_value) ;
    if isnumeric(file_name)
      return
    end
    new_value = fullfile(path_name, file_name);
    ValidateSetFile(src,evt,fn,new_value);
  end

  function ValidateSetFile(src,evt,fn,new_value)
    if nargin < 4,
      new_value = src.String;
    else
      src.String = new_value;
    end
    if ~exist(new_value,'file'),
      src.String = res.(fn);
      warndlg(sprintf('Singularity image %s does not exist',new_value),'Bad value','modal');
      return;
    end
    res.(fn) = new_value;
  end

  function tfsucc = Apply(varargin)
    tfsucc = true;
    for j = 1:numel(fns),
      try
        lObj.set_backend_property(fns{j},res.(fns{j}));
      catch ME,
        warndlg(sprintf('Error setting %s: %s',fns{j},getReport(ME)),'Error setting value','modal');
        res.(fns{j}) = lObj.get_backend_property(fns{j});
        val = res.(fns{j});
        if ~ischar(val),
          val = num2str(res.(fns{j}));
        end
        hs.(fns{j}).String = val;
        tfsucc = false;
      end

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
    delete(hdlg);
  end

end