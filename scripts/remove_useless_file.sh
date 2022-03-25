# !/bin/bash

echo "This cleans the folder /music-decomp/data"
echo "Please press ctrl-c if you did not run this in /music-decomp!!!"
printf "Press Enter if you are in /music-decomp"
read

# Removes in sub folder
for i in $(find -regex "^\.\/data\/.*/\..*$")
do 
    rm -v $i
done

# Removes in folder
for i in $(find -regex "^\.\/data/\..*$")
do 
    rm -v $i
done