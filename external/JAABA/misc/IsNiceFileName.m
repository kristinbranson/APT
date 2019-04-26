function v = IsNiceFileName(name)

v = ~isempty(regexp(name,'^[a-zA-Z][\w_\.]*$','once','start'));