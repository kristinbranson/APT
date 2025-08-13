function t = parseConfigJson(jsonfile)

s = TrnPack.hlpLoadJson(jsonfile);
t = TreeNode(s,true);