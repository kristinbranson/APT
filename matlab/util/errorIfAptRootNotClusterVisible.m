function errorIfAptRootNotClusterVisible(aptRootNativeAsChar)
% Error unless aptRootNativeAsChar is under /groups or /nrs.
%
% The bsub backend runs the Python deep-learning code (deepnet/APT_interface.py)
% from APT.Root on a cluster compute node.  Those nodes can only see the network
% filesystems /groups and /nrs, not a workstation's local disk, so if APT is
% checked out on local disk a bsub job fails on the node with an obscure
% "python: can't open file '.../deepnet/APT_interface.py'" error.  Calling this
% at bsub submission time turns that into a clear, immediate error instead.
if ~startsWith(aptRootNativeAsChar, {'/groups/', '/nrs/'})
  error('APT:aptRootNotClusterVisible', ...
        ['This copy of APT is at\n  %s\n' ...
         'but the bsub backend runs on cluster compute nodes, which can only see\n' ...
         'the /groups and /nrs network filesystems.  Run APT from a checkout under\n' ...
         '/groups or /nrs to use the bsub backend.'], ...
        aptRootNativeAsChar) ;
end
end  % function
