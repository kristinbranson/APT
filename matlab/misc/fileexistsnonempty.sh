#!/bin/bash
if [ -s $1 ]
then
    if [ $# -eq 1 ];
    then 
	echo "y"
    else
	# check size if size is supplied
        fsz=$(stat -c%s "$1")
	if [ $2 -eq $fsz ];
	then
	   echo "y"
	else
	   echo "n"
	fi
    fi
else
    echo "n"
fi
