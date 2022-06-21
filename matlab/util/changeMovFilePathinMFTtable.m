function outtbl = changeMovFilePathinMFTtable(intbl,datarootdir)
% need to test on within mac/linux, mac to windows, within windows
% works on windows to mac/linux

outtbl = intbl;

for i = 1:height(intbl)
    
    if ~ispc && any(strfind(intbl.mov{i},'\'))
        currmovpath = regexprep(intbl.mov{i},'\','/');
        [moviepath,moviename,c] = fileparts(currmovpath);
        [~,expdir] = fileparts(moviepath);
        newmovpath = fullfile(datarootdir,expdir,[moviename,c]);
        
    elseif ispc && any(strfind(intbl.mov{i},'/'))
        currmovpath = regexprep(intbl.mov{i},'/','\');
        [moviepath,moviename,c] = fileparts(currmovpath);
        [~,expdir] = fileparts(moviepath);
        newmovpath = fullfile(datarootdir,expdir,[moviename,c]);
        
    else
        [currmovpath,b,c] = fileparts(intbl.mov{i});
        [~,currmoviename] = fileparts(currmovpath);
        newmovpath = fullfile(datarootdir,currmoviename,[b,c]);
    end
    outtbl.mov{i} = newmovpath;
end
