function out = parseunits(in)

% empty case
if isempty(in),
  out.num = {'unit'};
  out.den = {};
  return;
end

% allowed unit names
knownunitnames = {'mm','cm','deg','rad','frame','sec','unit','px'};

% remove white space
in = strrep(in,' ','');

% find starts of unit names
timesidx = strfind(in,'*');
divideidx = strfind(in,'/');
% first unit is by default times
if ~ismember(1,timesidx) && ~ismember(1,divideidx),
  timesidx = [0,timesidx];
end
% also add on one character after end of string
allidx = [sort([timesidx,divideidx]),length(in)+1];
% take out the unit names separated by 
out.num = cell(1,length(timesidx));
out.den = cell(1,length(divideidx));
for i = 1:length(timesidx),
  i0 = timesidx(i)+1;
  i1 = allidx(find(allidx>i0,1))-1;
  out.num{i} = lower(in(i0:i1));
end
for i = 1:length(divideidx),
  i0 = divideidx(i)+1;
  i1 = allidx(find(allidx>i0,1))-1;
  out.den{i} = lower(in(i0:i1));
end

% a little flexibility
dictionary.mms = 'mm';
dictionary.millimeter = 'mm';
dictionary.millimeters = 'mm';
dictionary.cms = 'cm';
dictionary.centimeter = 'cm';
dictionary.centimeters = 'cm';
dictionary.degs = 'deg';
dictionary.degree = 'deg';
dictionary.degrees = 'deg';
dictionary.rads = 'rad';
dictionary.radian = 'rad';
dictionary.radians = 'rad';
dictionary.f = 'fr';
dictionary.fs = 'fr';
dictionary.frs = 'fr';
dictionary.frame = 'fr';
dictionary.frames = 'fr';
dictionary.second = 's';
dictionary.seconds = 's';
dictionary.sec = 's';
dictionary.secs = 's';
dictionary.units = 'unit';
dictionary.pxs = 'px';
dictionary.pixel = 'px';
dictionary.pixels = 'px';
flexiblenames = fieldnames(dictionary);
for i = 1:length(out.num),
  if ~ismember(out.num{i},knownunitnames) && ismember(out.num{i},flexiblenames),
    out.num{i} = dictionary.(out.num{i});
  end
end
for i = 1:length(out.den),
  if ~ismember(out.den{i},knownunitnames) && ismember(out.den{i},flexiblenames),
    out.den{i} = dictionary.(out.den{i});
  end
end