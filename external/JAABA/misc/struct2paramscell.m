function p = struct2paramscell(s)

p = [fieldnames(s),struct2cell(s)]';
p = p(:)';