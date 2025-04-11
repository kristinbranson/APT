function result = DLBackEndFromString(str)

if strcmpi(str,'docker')
  result = DLBackEnd.Docker;
elseif strcmpi(str,'bsub')
  result = DLBackEnd.Bsub;
elseif strcmpi(str,'conda')
  result = DLBackEnd.Conda;
elseif strcmpi(str,'aws')
  result = DLBackEnd.AWS;
else
  error('Could not convert string %s to a DLBackEnd value', str) ;
end
