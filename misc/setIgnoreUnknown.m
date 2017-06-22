function setIgnoreUnknown(h,varargin)

if mod(numel(varargin),2)==1
  error('setIgnoreUnknown:arg','Expected even number of arguments (p-v pairs)');
end

for i=1:2:numel(varargin)
  prop = varargin{i};
  val = varargin{i+1};
  try
    set(h,prop,val);
  catch ME
%     warningNoTrace('setIgnoreUnknown:unk',...
%       'Ignoring property set on ''%s'': %s',prop,ME.message);
  end
end