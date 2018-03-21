function whereisjavaclassloadingfrom(ClassName)
%WHEREISJAVACLASSLOADINGFROM  Show where a Java class is loaded from
%
% whereisjavaclassloadingfrom(ClassName)
%
% Shows where a Java class is loaded from in this Matlab session's JVM.
% This is for diagnosing Java class load problems, such as classpath
% ordering issues, seeing if a class of a given name is included in an
% unexpected JAR file, etc.
%
% Displays output to console.
%
% Examples:
%
% whereisjavaclassloadingfrom('java.util.HashMap')
% whereisjavaclassloadingfrom('com.ldhenergy.etools.MxUtil')
% whereisjavaclassloadingfrom('com.google.common.collect.Maps')
% whereisjavaclassloadingfrom('org.apache.commons.math.complex.Complex')

% Use javaArray to get Class object without having to instantiate. This
% lets it work with objects that have private or non-zero-arg constructors,
% and avoids side effects of object construction.
% (Would use java.lang.Class.forName(), because that's a more direct way of
% doing this, but it doesn't work for stuff on the dynamic classpath.)
ja = javaArray(ClassName,1);
klass = ja.getClass().getComponentType();

klassLoader = klass.getClassLoader();
if isempty(klassLoader)
    % JVM used null to represent the "bootstrap" class loader
    % I think that's the same as the "system" class loader
    klassLoader = java.lang.ClassLoader.getSystemClassLoader();
end
klassLoaderStr = char(klassLoader.toString());

klassFilePath = [strrep(ClassName, '.', '/') '.class'];
try
    % This logic assumes that the classes exist as files in the class
    % loader. It's a valid assumption for mainstream class loaders,
    % including the one's I've seen with Matlab.
    klassUrl = klassLoader.getResource(klassFilePath);
    if isempty(klassUrl)
        klassUrlStr = '';
    else
        klassUrlStr = char(klassUrl.toString());
    end
catch err
    klassUrlStr = sprintf('ERROR: %s', err.message);
end

% Get all locations, to reveal masked definitions
urls = enumeration2array(klassLoader.getResources(klassFilePath));

disp(sprintf('Version: %s\nClass:       %s\nClassLoader: %s\nURL:         %s', version,...
    char(klass.getName()), klassLoaderStr, klassUrlStr));
if numel(urls) > 1
    disp('Class is masked:');
    for i = 1:numel(urls)
        disp(sprintf('URL %d:       %s', i, char(urls(i))));
    end
end

%%
function out = enumeration2array(jenum)
tmp = {};
while jenum.hasMoreElements()
    tmp{end+1} = jenum.nextElement();
end
out = [tmp{:}];