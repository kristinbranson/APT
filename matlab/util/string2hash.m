%STRING2HASH Convert a string to a 64 char hex hash string (256 bit hash)
%
%   hash = string2hash(string)
%
%IN:
%   string - a string!
%
%OUT:
%   hash - a 64 character string, encoding the 256 bit SHA hash of string
%          in hexadecimal. 
function hash = string2hash(string)
persistent md
if isempty(md)
    md = java.security.MessageDigest.getInstance('SHA-256');
end
hash = sprintf('%2.2x', typecast(md.digest(uint8(string)), 'uint8')');
end