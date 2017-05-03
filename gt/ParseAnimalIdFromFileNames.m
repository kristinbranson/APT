function animalids = ParseAnimalIdFromFileNames(moviefiles,datatype)

switch datatype,
  case 'stephen',
    animalids = regexp(moviefiles,'fly_?(\d+)[^\d]','tokens');
    assert(~any(cellfun(@isempty,animalids)));
    animalids = cellfun(@(x) str2double(x{end}),animalids);
  case 'jan'
    animalids = regexp(moviefiles,'(\d{6})_(\d{2})','tokens');
    assert(~any(cellfun(@isempty,animalids)));
    animalids = cellfun(@(x) str2double([x{end}{1},x{end}{2}]),animalids);
  case 'roian'
    m = regexp(moviefiles,'experiment_?(?<expnum>\d+)_(?<date>\d{8})T(?<time>\d{6}).*(?<segnum>\d+)_[mM]ouse_?(?<mouse>\d+)\.','once','names');
    assert(~any(cellfun(@isempty,m)));
    animalids = cellfun(@(x) str2double(sprintf('%02d%02d%02d%08d%06d',str2double(x.expnum),str2double(x.segnum),str2double(x.mouse),str2double(x.date),str2double(x.time))),m);
  case 'jay'
    animalids = regexp(moviefiles,'M(\d+)[^\d]','tokens');
    assert(~any(cellfun(@isempty,animalids)));
    animalids = cellfun(@(x) str2double(x{end}),animalids);   
  case 'romain'
    m = regexp(moviefiles,'date_(\d{4})_(\d{2})_(\d{2})_time_(\d{2})_(\d{2})_(\d{2})_v\d+\.avi','tokens','once');
    assert(~any(cellfun(@isempty,m)));
    animalids = cellfun(@(x) str2double([x{:}]),m);
  otherwise
    error('not implemented');
    
end

assert(all(~isnan(animalids)));

