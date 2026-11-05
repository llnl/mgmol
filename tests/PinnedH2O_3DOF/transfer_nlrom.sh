#!/bin/bash

ncpus=8
target_dir="/usr/workspace/nlrom/MGmol/PinnedH2O_3DOF/data_${ncpus}"

if [ ! -d $target_dir ]; then
  mkdir -p $target_dir
fi

for dir in data/*/; do
  dirname=$(basename "$dir")
  new_dir="$target_dir/${dirname}"
  if [ ! -d $new_dir ]; then
    mkdir -p $new_dir
  fi
  mv -f $dir/* $new_dir/
  rm -rf $dir
done

rm -rf data
