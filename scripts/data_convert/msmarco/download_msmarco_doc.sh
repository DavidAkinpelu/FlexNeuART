#!/bin/bash -e
# Downloading and re-packing the MSMARCO passage collection.
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source scripts/common_proc.sh

cd "$dstDir"



for fn in msmarco-docdev-qrels.tsv.gz \
          msmarco-docs.tsv.gz	\
          msmarco-doctrain-queries.tsv.gz \
          msmarco-docdev-queries.tsv.gz \
          msmarco-doctrain-qrels.tsv.gz \
          msmarco-test2019-queries.tsv.gz \
          msmarco-test2020-queries.tsv.gz \
          ; do
  uri=https://msmarco.blob.core.windows.net/msmarcoranking/$fn
  echo "Downloading $uri"
  wget "$uri"
done

for year in 2019 2020 ; do
  gunzip msmarco-test${year}-queries.tsv.gz
done

wget https://trec.nist.gov/data/deep/2019qrels-docs.txt