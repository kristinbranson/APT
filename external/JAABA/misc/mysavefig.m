function mysavefig(fname, varargin)
	
% Usage: savefig(filename, fighdl, format, options)
%
% Saves a pdf, eps, png, jpeg, and/or tiff of the contents of the fighandle's (or current) figure.
% It saves an eps of the figure and the uses Ghostscript to convert to the other formats.
% The result is a cropped, clean picture. There are options for using rgb or cmyk colours,
% or grayscale. You can also choose the resolution.
%
% The advantage of savefig is that there is very little empty space around the figure in the
% resulting files, you can export to more than one format at once, and Ghostscript generates
% trouble-free files.
%
% If you find any errors, please let me know! (peder at axensten dot se)
%
% filename: File name without suffix.
%
% fighdl:  (default: gcf) Integer handle to figure.
%
% options: (default: '-r300', '-lossless', '-rgb') You can define your own
%          defaults in a global variable savefig_defaults, if you want to, i.e.
%          savefig_defaults= {'-r200','-gray'};.
% 'eps':   Output in Encapsulated Post Script (no preview yet).
% 'pdf':   Output in (Adobe) Portable Document Format.
% 'png':   Output in Portable Network Graphics.
% 'jpeg':  Output in Joint Photographic Experts Group format.
% 'tiff':  Output in Tagged Image File Format (no compression: huge files!).
% '-rgb':  Output in rgb colours.
% '-cmyk': Output in cmyk colours (not yet 'png' or 'jpeg' -- '-rgb' is used).
% '-gray': Output in grayscale (not yet 'eps' -- '-rgb' is used).
% '-lossless':  Use lossless compression, works on most formats. same as '-c0', below.
% '-c<float>':  Set compression for non-indexed bitmaps in PDFs -
%               0: lossless; 0.1: high quality; 0.5: medium; 1: high compression.
% '-r<integer>':  Set resolution.
% '-crop': Removes points and line segments outside the viewing area --
% permanently.
%          Only use this on figures where many points and/or line segments are outside
%          the area zoomed in to. This option will result in smaller vector files (has no
%          effect on pixel files).
% '-dbg':  Displays gs command line(s).
%
% EXAMPLE:
% savefig('nicefig', 'pdf', 'jpeg', '-cmyk', '-c0.1', '-r250');
% Saves the current figure to nicefig.pdf and nicefig.png, both in cmyk and at 250 dpi,
%          with high quality lossy compression.
%
% REQUIREMENT: Ghostscript. Version 8.57 works, probably older versions too, but '-dEPSCrop'
%          must be supported. I think version 7.32 or newer is ok.
%
% HISTORY:
% Version 1.0, 2006-04-20.
% Version 1.1, 2006-04-27:
% - No 'epstopdf' stuff anymore! Using '-dEPSCrop' option in gs instead!
% Version 1.2, 2006-05-02:
% - Added a '-dbg' option (see options, above).
% - Now looks for a global variable 'savefig_defaults' (see options, above).
% - More detailed Ghostscript options (user will not really notice).
% - Warns when there is no device for a file-type/color-model combination.
% Version 1.3, 2006-06-06:
% - Added a check to see if there actually is a figure handle.
% - Now works in Matlab 6.5.1 (R13SP1) (maybe in 6.5 too).
% - Now compatible with Ghostscript 8.54, released 2006-06-01.
% Version 1.4, 2006-07-20:
% - Added an option '-soft' that enables anti-aliasing on pixel graphics (on by default).
% - Added an option '-hard' that don't do anti-aliasing on pixel graphics.
% Version 1.5, 2006-07-27:
% - Fixed a bug when calling with a figure handle argument.
% Version 1.6, 2006-07-28:
% - Added a crop option, see above.
% Version 1.7, 2007-03-31:
% - Fixed bug: calling print with invalid renderer value '-none'.
% - Removed GhostScript argument '-dUseCIEColor' as it sometimes discoloured things.
% Version 1.8, 2008-01-03:
% - Added MacIntel: 'MACI'.
% - Added 64bit PC (I think, can't test it myself).
% - Added option '-nointerpolate' (use it to prevent blurring of pixelated).
% - Removed '-hard' and '-soft'. Use '-nointerpolate' for '-hard', default for '-soft'.
% - Fixed the gs 8.57 warning on UseCIEColor (it's now set).
% - Added '-gray' for pdf, but gs 8.56 or newer is needed.
% - Added '-gray' and '-cmyk' for eps, but you a fairly recent gs might be needed.
% Version 1.9, 2008-07-27:
% - Added lossless compression, see option '-lossless', above. Works on most formats.
% - Added lossy compression, see options '-c<float>...', above. Works on 'pdf'.
%   Thanks to Olly Woodford for idea and implementation!
% - Removed option '-nointerpolate' -- now savefig never interpolates.
% - Fixed a few small bugs and removed some mlint comments. 
%
% TO DO: (Need Ghostscript support for these, so don't expect anything soon...)
% - svg output.
% - '-cmyk' for 'jpeg' and 'png'.
% - Preview in 'eps'.
% - Embedded vector fonts, not bitmap, in 'eps'.
%
% Copyright (C) Peder Axensten (peder at axensten dot se), 2006.

% KEYWORDS:     eps, pdf, jpg, jpeg, png, tiff, eps2pdf, epstopdf, ghostscript
%
% INSPIRATION:  eps2pdf (5782), eps2xxx (6858)
%
% REQUIREMENTS: Works in Matlab 6.5.1 (R13SP1) (maybe in 6.5 too).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
  % parse my parameters
  doremove = false(1,length(varargin));
  fontname = [];
  fontsize = [];
  fontmode='scaled';
  width = -1;
  height = -1;
  i = 0;
  while true,
    i = i + 1;
    if i >= length(varargin), break; end
    if ~ischar(varargin{i}), continue; end
    switch lower(varargin{i}),
      case 'fontmode'
    fontmode = lower(varargin{i+1});
    if (~strcmp(fontmode,{'scaled','fixed'}))
      error('FontMode must be ''scaled'' or ''fixed''.');
    end
      case 'fontname'
        fontname = varargin{i+1};
        if ~any(strcmpi(fontname,listfonts)),
          error('FontName must be a member of listfonts');
        end
        doremove([i,i+1]) = true;
        i = i+1;
      case 'fontsize'
        fontsize = LocalToNum(varargin{i+1});
        if(~LocalIsPositiveScalar(fontsize))
          error('FontSize must be a numeric scalar > 0');
        end
        doremove([i,i+1]) = true;
        i = i+1;
      case 'width',
        width = LocalToNum(varargin{i+1});
        if(~LocalIsPositiveScalar(width))
          error('Width must be a numeric scalar > 0');
        end
        doremove([i,i+1]) = true;
        i = i+1;
      case 'height',
          height = LocalToNum(varargin{i+1});
          if(~LocalIsPositiveScalar(height))
            error('Height must be a numeric scalar > 0');
          end
        doremove([i,i+1]) = true;
        i = i+1;
    end
  end
  varargin(doremove) = [];
  realnargin = nargin - nnz(doremove);
  
  % Get handle.
	fighdl=		get(0, 'CurrentFigure'); % See gcf.						% Get figure handle.
	if((realnargin >= 2) && (numel(varargin{1}) == 1) && isnumeric(varargin{1}))
		fighdl=		varargin{1};
		varargin=	{varargin{2:end}};
	end
	if(isempty(fighdl)), error('There is no figure to save!?');	end
  H = fighdl;
  
  allText   = findall(H, 'type', 'text');
  allAxes   = findall(H, 'type', 'axes');
  allFont   = [allText; allAxes];

  old.objs = {};
  old.prop = {};
  old.values = {};
  
  % Process size parameters
  paperPos = get(H, 'PaperPosition');
  old = LocalPushOldData(old, H, 'PaperPosition', paperPos);
  figureUnits = get(H, 'Units');
  set(H, 'Units', get(H,'PaperUnits'));
  figurePos = get(H, 'Position');
  aspectRatio = figurePos(3)/figurePos(4);
  set(H, 'Units', figureUnits);
  if (width == -1) & (height == -1)
    width = paperPos(3);
    height = paperPos(4);
  elseif (width == -1)
    width = height * aspectRatio;
  elseif (height == -1)
    height = width / aspectRatio;
  end
  set(H, 'PaperPosition', [0 0 width height]);
  paperPosMode = get(H, 'PaperPositionMode');
  old = LocalPushOldData(old, H, 'PaperPositionMode', paperPosMode);
  set(H, 'PaperPositionMode', 'manual');
  oldPosition = get(H,'Position');
  newPosition = oldPosition;
  newPosition(3) = width;
  newPosition(4) = height;
  set(H,'Position',newPosition);
  old = LocalPushOldData(old,H,'Position',oldPosition);
  
  % Process font parameters
  if ~isempty(fontname)
    set(allFont,'fontname',fontname);
  end
  if (~isempty(fontmode))
    oldfonts = LocalGetAsCell(allFont,'FontSize');
    switch (fontmode)
      case 'fixed'
        oldfontunits = LocalGetAsCell(allFont,'FontUnits');
        old = LocalPushOldData(old, allFont, {'FontUnits'}, oldfontunits);
        set(allFont,'FontUnits','points');
        if (fontsize == -1)
          set(allFont,'FontSize',7);
        else
          set(allFont,'FontSize',fontsize);
        end
      case 'scaled'
        if (fontsize == -1)
          wscale = width/figurePos(3);
          hscale = height/figurePos(4);
          scale = min(wscale, hscale);
        else
          scale = 1;
        end
        newfonts = LocalScale(oldfonts,scale,5);
        set(allFont,{'FontSize'},newfonts);
      otherwise
        error('Invalid FontMode parameter');
    end
    % make sure we push the size after the units
    old = LocalPushOldData(old, allFont, {'FontSize'}, oldfonts);
  end
  
	op_dbg=		false;												% Default value.
	
	% Compression
	compr=			[' -dUseFlateCompression=true -dLZWEncodePages=true -dCompatibilityLevel=1.6' ...
					 ' -dAutoFilterColorImages=false -dAutoFilterGrayImages=false ' ...
					 ' -dColorImageFilter=%s -dGrayImageFilter=%s'];	% Compression.
	lossless=		sprintf (compr, '/FlateEncode', '/FlateEncode');
	lossy=			sprintf (compr, '/DCTEncode',   '/DCTEncode'  );
	lossy=			[lossy ' -c ".setpdfwrite << /ColorImageDict << /QFactor %g ' ...
					 '/Blend 1 /HSample [%s] /VSample [%s] >> >> setdistillerparams"'];
	
	% Create gs command.
	cmdEnd=			' -sDEVICE=%s -sOutputFile="%s"';					% Essential.
	epsCmd=			'';
	epsCmd=	[epsCmd ' -dSubsetFonts=true -dEmbedAllFonts=false -dNOPLATFONTS'];% Future support?
	epsCmd=	[epsCmd ' -dUseCIEColor=true -dColorConversionStrategy=/UseDeviceIndependentColor' ...
					' -dProcessColorModel=/%s'];						% Color conversion.
	pdfCmd=	[epsCmd ' -dAntiAliasColorImages=false' cmdEnd];
	epsCmd=	[epsCmd cmdEnd];
	
	% Get file name.
	if((realnargin < 1) || isempty(fname) || ~ischar(fname))				% Check file name.
		error('No file name specified.');
	end
	[pathstr, namestr] = fileparts(fname);
	if(isempty(pathstr)), fname= fullfile(cd, namestr);	end
	
	set(fighdl, 'Units', 'centimeters')									% Set paper stuff.
	sz=			get(fighdl, 'Position');
	sz(1:2)=	0;
	set(fighdl, 'PaperUnits', 'centimeters', 'PaperSize', sz(3:4), 'PaperPosition', sz);
	
	% Set up the various devices.
	% Those commented out are not yet supported by gs (nor by savefig).
	% pdf-cmyk works due to the Matlab '-cmyk' export being carried over from eps to pdf.
	device.eps.rgb=		sprintf(epsCmd, 'DeviceRGB',	'epswrite', [fname '.eps']);
	device.jpeg.rgb=	sprintf(cmdEnd,	'jpeg', 					[fname '.jpeg']);
%	device.jpeg.cmyk=	sprintf(cmdEnd,	'jpegcmyk', 				[fname '.jpeg']);
	device.jpeg.gray=	sprintf(cmdEnd,	'jpeggray',					[fname '.jpeg']);
	device.pdf.rgb=		sprintf(pdfCmd, 'DeviceRGB',	'pdfwrite', [fname '.pdf']);
	device.pdf.cmyk=	sprintf(pdfCmd, 'DeviceCMYK',	'pdfwrite', [fname '.pdf']);
	device.pdf.gray=	sprintf(pdfCmd, 'DeviceGray',	'pdfwrite', [fname '.pdf']);
	device.png.rgb=		sprintf(cmdEnd,	'png16m', 					[fname '.png']);
%	device.png.cmyk=	sprintf(cmdEnd,	'png???', 					[fname '.png']);
	device.png.gray=	sprintf(cmdEnd,	'pnggray', 					[fname '.png']);
	device.tiff.rgb=	sprintf(cmdEnd,	'tiff24nc',					[fname '.tiff']);
	device.tiff.cmyk=	sprintf(cmdEnd,	'tiff32nc', 				[fname '.tiff']);
	device.tiff.gray=	sprintf(cmdEnd,	'tiffgray', 				[fname '.tiff']);
	
	% Get options.
	global savefig_defaults;											% Add global defaults.
 	if( iscellstr(savefig_defaults)), varargin=	{savefig_defaults{:}, varargin{:}};
	elseif(ischar(savefig_defaults)), varargin=	{savefig_defaults, varargin{:}};
	end
	varargin=	{'-r300', '-lossless', '-rgb', varargin{:}};			% Add defaults.
	res=		'';
	types=		{};
	crop=		false;
	for n= 1:length(varargin)											% Read options.
		if(ischar(varargin{n}))
			switch(lower(varargin{n}))
			case {'eps','jpeg','pdf','png','tiff'},		types{end+1}=	lower(varargin{n});
			case '-rgb',				color=	'rgb';	deps= {'-depsc2'};
			case '-cmyk',				color=	'cmyk';	deps= {'-depsc2', '-cmyk'};
			case '-gray',				color=	'gray';	deps= {'-deps2'};
			case '-lossless',			comp=			0;
			case '-crop',				crop=			true;
			case '-dbg',				op_dbg=			true;
			otherwise
				if(regexp(varargin{n}, '^\-r[0-9]+$')), 	 res=  varargin{n};
				elseif(regexp(varargin{n}, '^\-c[0-9.]+$')), comp= str2double(varargin{n}(3:end));
				else	warning('pax:savefig:inputError', 'Unknown option in argument: ''%s''.', varargin{n});
				end
			end
		else
			warning('pax:savefig:inputError', 'Wrong type of argument: ''%s''.', class(varargin{n}));
		end
	end
	types=		unique(types);
	if(isempty(types)), error('No output format given.');	end
	
	if (comp == 0)														% Lossless compression
		gsCompr=		lossless;
	elseif (comp <= 0.1)												% High quality lossy
		gsCompr=		sprintf(lossy, comp, '1 1 1 1', '1 1 1 1');
	else																% Normal lossy
		gsCompr=		sprintf(lossy, comp, '2 1 1 2', '2 1 1 2');
	end
	
	% Generate the gs command.
	switch(computer)													% Get gs command.
		case {'MAC','MACI'},			gs= '/usr/local/bin/gs';
		case {'PCWIN','PCWIN64'},		gs= 'gswin32c.exe';
		otherwise,						gs= 'gs';
	end
	gs=		[gs		' -q -dNOPAUSE -dBATCH -dEPSCrop'];					% Essential.
	gs=		[gs		' -dUseFlateCompression=true'];						% Useful stuff.
	gs=		[gs		' -dAutoRotatePages=/None'];						% Probably good.
	gs=		[gs		' -dHaveTrueTypes'];								% Probably good.
	gs=		[gs		' ' res];											% Add resolution to cmd.
	
	if(crop && ismember(types, {'eps', 'pdf'}))							% Crop the figure.
		fighdl= do_crop(fighdl);
	end
	
	% Output eps from Matlab.
	renderer=	['-' lower(get(fighdl, 'Renderer'))];					% Use same as in figure.
	if(strcmpi(renderer, '-none')), renderer=	'-painters';	end		% We need a valid renderer.
	print(fighdl, deps{:}, '-noui', renderer, res, [fname '-temp']);	% Output the eps.
	
	% Convert to other formats.
	for n= 1:length(types)												% Output them.
		if(isfield(device.(types{n}), color))
			cmd=		device.(types{n}).(color);						% Colour model exists.
		else
			cmd=		device.(types{n}).rgb;							% Use alternative.
			if(~strcmp(types{n}, 'eps'))	% It works anyways for eps (VERY SHAKY!).
				warning('pax:savefig:deviceError', ...
						'No device for %s using %s. Using rgb instead.', types{n}, color);
			end
		end
		cmp=	lossless;
		if (strcmp(types{n}, 'pdf')),	cmp= gsCompr;		end			% Lossy compr only for pdf.
		if (strcmp(types{n}, 'eps')),	cmp= '';			end			% eps can't use lossless.
		cmd=	sprintf('%s %s %s -f "%s-temp.eps"', gs, cmd, cmp, fname);% Add up.
			status= system(cmd);										% Run Ghostscript.
			if (op_dbg || status), 		display (cmd),		end
	end
	delete([fname '-temp.eps']);										% Clean up.

  % Restore figure settings
  for n = 1:length(old.objs)
    set(old.objs{n}, old.prop{n}, old.values{n});
  end
  
end % ends savefig function

function fig= do_crop(fig)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%	Remove line segments that are outside the view.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	haxes=	findobj(fig, 'Type', 'axes', '-and', 'Tag', '');
	for n=1:length(haxes)
		xl=		get(haxes(n), 'XLim');
		yl=		get(haxes(n), 'YLim');
		lines=	findobj(haxes(n), 'Type', 'line');
		for m=1:length(lines)
			x=				get(lines(m), 'XData');
			y=				get(lines(m), 'YData');
			
			inx=			(xl(1) <= x) & (x <= xl(2));	% Within the x borders.
			iny=			(yl(1) <= y) & (y <= yl(2));	% Within the y borders.
			keep=			inx & iny;						% Within the box.
			
			if(~strcmp(get(lines(m), 'LineStyle'), 'none'))
				crossx=		((x(1:end-1) < xl(1)) & (xl(1) < x(2:end))) ...	% Crossing border x1.
						|	((x(1:end-1) < xl(2)) & (xl(2) < x(2:end))) ...	% Crossing border x2.
						|	((x(1:end-1) > xl(1)) & (xl(1) > x(2:end))) ...	% Crossing border x1.
						|	((x(1:end-1) > xl(2)) & (xl(2) > x(2:end)));	% Crossing border x2.
				crossy=		((y(1:end-1) < yl(1)) & (yl(1) < y(2:end))) ...	% Crossing border y1.
						|	((y(1:end-1) < yl(2)) & (yl(2) < y(2:end))) ...	% Crossing border y2.
						|	((y(1:end-1) > yl(1)) & (yl(1) > y(2:end))) ...	% Crossing border y1.
						|	((y(1:end-1) > yl(2)) & (yl(2) > y(2:end)));	% Crossing border y2.
				crossp=	[(	(crossx & iny(1:end-1) & iny(2:end)) ...	% Crossing a x border within y limits.
						|	(crossy & inx(1:end-1) & inx(2:end)) ...	% Crossing a y border within x limits.
						|	crossx & crossy ...							% Crossing a x and a y border (corner).
						 ),	false ...
						];
				crossp(2:end)=	crossp(2:end) | crossp(1:end-1);		% Add line segment's secont end point.
			
				keep=			keep | crossp;
			end
			set(lines(m), 'XData', x(keep))
			set(lines(m), 'YData', y(keep))
		end
	end
end

function outData = LocalPushOldData(inData, objs, prop, values)
outData.objs = {inData.objs{:}, objs};
outData.prop = {inData.prop{:}, prop};
outData.values = {inData.values{:}, values};
end

function cellArray = LocalGetAsCell(fig,prop)
cellArray = get(fig,prop);
if (~isempty(cellArray)) & (~iscell(cellArray))
  cellArray = {cellArray};
end
end

function newArray = LocalScale(inArray, scale, minValue)
n = length(inArray);
newArray = cell(n,1);
for k=1:n
  newArray{k} = max(minValue,scale*inArray{k}(1));
end
end

function bool = LocalIsPositiveScalar(value)
bool = isnumeric(value) & ...
       prod(size(value)) == 1 & ...
       value > 0;
end

function value = LocalToNum(value)
if ischar(value)
  value = str2num(value);
end
end