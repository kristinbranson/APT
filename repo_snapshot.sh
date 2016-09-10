#!/bin/bash
 
scriptpath="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

nocolor=false
brief=false

while [[ $# > 0 ]]
do
key="$1"

case $key in
    -nocolor)
    nocolor=true
    ;;
    -brief)
    brief=true
    ;;
    *)

    break
    ;;
esac
shift
done

if [ "$#" -eq 0 ]; then 
    DIR=$scriptpath
elif [ "$#" -eq 1 ]; then
    DIR=$1
else
    echo "Usage: $0 [-nocolor] [-brief] [repodir]"
    exit 1
fi

if [ ! -d "$DIR" ]; then 
    echo "Error: directory $DIR does not exist"
    exit 1
fi

pushd . >/dev/null
cd $DIR
echo $DIR
if [ $brief = true ]; then
    headn=2
else
    headn=10
fi
if [ $nocolor = true ]; then
    git log --graph --full-history --all --pretty=format:"%h%x09%d%x20%s" | head -n $headn
else
    "$scriptpath/git-graph.sh" | head -n $headn
fi
git status --porcelain .
popd >/dev/null