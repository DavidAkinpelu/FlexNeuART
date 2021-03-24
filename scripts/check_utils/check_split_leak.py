#!/usr/bin/env python
import sys
import os
import json
import argparse
import random
import math
import pytorch_pretrained_bert
import time
import numpy as np
from tqdm import tqdm

#
# This utility scripts checks for possible leakage across different data splits.
# Importantly it works only for bitext. In the context of a community QA collection,
# such bitext arises naturally. For regular document collections, a pseudo-bitext
# needs to be created user the scripts/giza/export_bitext_plain.sh:
# importantly one needs to use the text_raw field (and text as an index field)
# and use 0 for the  "max query to doc word ratio" so documents are not split
# into chunks.
#

# Specifically, we search for very similar question-answer pairs, which might
# be duplicates or near duplicates. Hence, we check the following:
# 1. Are there very similar questions?
# 2. For sufficiently similar questions, e.g., Jaccard >= 0.75, we check
#    all pairwise similarities among all relevant answers.
#
# By default this method uses brute-force search with the Jaccard similarity.
# The exhaustiveness of the search ensures we won't miss anything. However, for quicker-and-easier
# checks, one can use HNSW with sufficently high values of M (>= 30), efConstruction (>=200),
# and efSearch (>=1000). These parameters might need to be bumped up for "harder" collections
# and brute-force search is certainly a safer option.
#

sys.path.append('.')

from scripts.check_utils.common_check import getTokenIds, QUERY_BATCH_SIZE, jaccard, \
                                            readSampleQueries, createJaccardIndex, strToNMSLIBVect

from scripts.data_convert.convert_common import jsonlGen, unique
from scripts.config import BERT_BASE_MODEL, \
                         ANSWER_FILE_JSON, QREL_FILE, \
                         DOCID_FIELD, TEXT_RAW_FIELD_NAME

from scripts.common_eval import readQrelsDict

PRINT_TOO_CLOSE_THRESHOLD=0.9 # We want to inspect answers that are too close

np.random.seed(0)

BERT_TOKENIZER = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

parser = argparse.ArgumentParser(description='Checking for possible high overlaps among QA pairs.')

parser.add_argument('--data_dir',
                    metavar='data directory',
                    help='data directory',
                    type=str, required=True)
parser.add_argument('--input_subdir1',
                    metavar='1st input subdir',
                    help='1st input data subdirectory',
                    type=str, required=True)
parser.add_argument('--input_subdir2',
                    metavar='1st input subdir',
                    help='1st input data subdirectory',
                    type=str, required=True)
parser.add_argument('--sample_prob1',
                    metavar='1st subdir sample prob',
                    type=float, default=1.0)
parser.add_argument('--sample_prob2',
                    metavar='2d subdir sample prob',
                    type=float, default=1.0)
parser.add_argument('-k', metavar='k-NN k',
                    type=int, default=1)
parser.add_argument('--min_jacc', metavar='min jaccard to compare answers',
                    type=float, default=1.0)
parser.add_argument("--use_hnsw", action="store_true",
                    help="Use HNSW instead of brute-force for retrieval")



args = parser.parse_args()
print(args)

dataDir = args.data_dir

sampleQueryList1, sampleQueryList2 = readSampleQueries(dataDir,
                                                       args.input_subdir1, args.sample_prob1,
                                                       args.input_subdir2, args.sample_prob2)

apath1=os.path.join(dataDir, args.input_subdir1, ANSWER_FILE_JSON)
apath2=os.path.join(dataDir, args.input_subdir2, ANSWER_FILE_JSON)

rpath1 = os.path.join(dataDir, args.input_subdir1, QREL_FILE)
qrelDict1 = readQrelsDict(rpath1)
print('Read %d qrel sets from %s' % (len(qrelDict1), rpath1))
rpath2 = os.path.join(dataDir, args.input_subdir2, QREL_FILE)
qrelDict2 = readQrelsDict(rpath2)
print('Read %d qrel sets from %s' % (len(qrelDict2), rpath2))

answDictText = {}

for fn in [apath1, apath2]:
    qty = 0

    for e in tqdm(jsonlGen(fn), desc='loading answers'):
        qty += 1

        answId = e[DOCID_FIELD]
        answText = e[TEXT_RAW_FIELD_NAME]

        answDictText[answId] = answText

    print('Read %d answers from %s' % (qty, fn))

index = createJaccardIndex(args.use_hnsw, BERT_TOKENIZER, sampleQueryList2)

K = args.k
print('K=', K)


nbrQuestSimils = []
nbrAnswSimils = []

for start in tqdm(range(0, len(sampleQueryList1), QUERY_BATCH_SIZE), desc='query w/ 1st query set'):
    qbatch = []
    for e in sampleQueryList1[start:start + QUERY_BATCH_SIZE]:
        qbatch.append(strToNMSLIBVect(BERT_TOKENIZER, e[TEXT_RAW_FIELD_NAME]))

    if qbatch:
        nbrs = index.knnQueryBatch(qbatch, k=K, num_threads=0)
        assert(len(nbrs))
        for i in range(len(qbatch)):
            qnum1 = start + i
            qid1 = sampleQueryList1[qnum1][DOCID_FIELD]

            indexQueries, dists = nbrs[i]
            for t in range(len(indexQueries)):
                # In the case of Jaccard, the similarity is one minus the distance
                nqsimil = 1 - dists[t]
                nbrQuestSimils.append(nqsimil)

                # For close enough queries, compute all pairwise distances
                # between the respective relevant answers
                if nqsimil >= args.min_jacc:
                    qnum2 = indexQueries[t]

                    qid2 = sampleQueryList2[qnum2][DOCID_FIELD]

                    if qid1 in qrelDict1 and qid2 in qrelDict2:
                        for aid1, grade1 in qrelDict1[qid1].items():
                            for aid2, grade2 in qrelDict2[qid2].items():
                                if grade1 > 0 and grade2 > 0 and \
                                    aid1 in answDictText and aid2 in answDictText:
                                    toks1 = unique(getTokenIds(BERT_TOKENIZER, answDictText[aid1]))
                                    toks2 = unique(getTokenIds(BERT_TOKENIZER, answDictText[aid2]))
                                    answSimil = jaccard(toks1, toks2)
                                    nbrAnswSimils.append(answSimil)
                                    if answSimil >= PRINT_TOO_CLOSE_THRESHOLD:
                                        print(qid1, aid1, '<=>', answSimil, '<=>', qid2, aid2)
                                        print('---------------------')
                                        print(answDictText[aid1])
                                        print(toks1)
                                        print('---------------------')
                                        print(answDictText[aid2])
                                        print(toks2)
                                        print('=====================')

        qbatch = []

# We are more interested in extremely high similarities, hence,
# we increase resolution in higher quantiles
q=list([0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.999])
q.sort()

print('Maximum similarity among questions:', np.max(nbrQuestSimils))
print('Distribution of question-neighbor *SIMILARITIES* for k=%d' % K)
dst = np.quantile(nbrQuestSimils, q = q)
print(' quant| simil')
print('------+------')
for k in range(len(q)):
    print('%5.03g' % q[k], ' | %.05g' % dst[k])

print('Distribution of relevant answer pairwise *SIMILARITIES* from neighbor questions with Jaccard >= %g' % args.min_jacc)
if nbrAnswSimils:
    dst = np.quantile(nbrAnswSimils, q = q)
    print(' quant| simil')
    print('------+------')
    for k in range(len(q)):
        print('%5.03g' % q[k], ' | %.05g' % dst[k])
else:
    print('No data collected, did you set the Jaccard threshold to a value < 1?')


print('Check is successful!')




