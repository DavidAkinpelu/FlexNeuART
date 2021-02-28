#!/usr/bin/env python
# A file to generate queries & qrels from the ORCAS data collection
#
import sys
import os
import json
import argparse
import pytorch_pretrained_bert

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, readStopWords, addRetokenizedField, readQueries, MAX_NUM_QUERY_OPT_HELP, MAX_NUM_QUERY_OPT

from scripts.config import TEXT_BERT_TOKENIZED_NAME, TEXT_UNLEMM_FIELD_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TEXT_RAW_FIELD_NAME, \
    REPORT_QTY, SPACY_MODEL, QUESTION_FILE_JSON, QREL_FILE

from scripts.common_eval import QrelEntry, writeQrels

ORCAS_QID_PREF='orcas_'

parser = argparse.ArgumentParser(description='Convert MSMARCO-ORCAS queries & qrels.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--filter_query_dir', metavar='filtering query dir',
                    default=[],
                    help=f'all queries found in {QUESTION_FILE_JSON} files from these directories are ignored',
                    nargs='*')
parser.add_argument('--out_dir', metavar='output directory', help='output directory',
                    type=str, required=True)
parser.add_argument('--min_query_token_qty', type=int, default=0,
                    metavar='min # of query tokens', help='ignore queries that have smaller # of tokens')
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)
parser.add_argument('--' + MAX_NUM_QUERY_OPT, type=int, default=None, help=MAX_NUM_QUERY_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inpFile = FileWrapper(args.input)
maxQueryQty = arg_vars[MAX_NUM_QUERY_OPT]
if maxQueryQty < 0 or maxQueryQty is None:
    maxQueryQty = float('inf')

ignoreQueries = set()

for qfile_dir in args.filter_query_dir:
    qfile_name = os.path.join(qfile_dir, QUESTION_FILE_JSON)
    for e in readQueries(qfile_name):
        ignoreQueries.add(e[TEXT_FIELD_NAME])
    print('Read queries from: ' + qfile_name)

print('A list of queries to ignore has %d entries' % (len(ignoreQueries)))


if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

outFileQueries = FileWrapper(os.path.join(args.out_dir, QUESTION_FILE_JSON), 'w')
outFileQrelsName = os.path.join(args.out_dir, QREL_FILE)

minQueryTokQty = args.min_query_token_qty

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

if arg_vars[BERT_TOK_OPT]:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

qrelList = []

# Input file is a TSV file
ln = 0

prevQid = ''
genQueryQty = 0

for line in inpFile:
    ln += 1
    line = line.strip()
    if not line:
        continue
    fields = line.split('\t')
    if len(fields) != 4:
        print('Misformated line %d ignoring:' % ln)
        print(line.replace('\t', '<field delimiter>'))
        continue

    qid_orig, query, did, _  = fields
    qid = ORCAS_QID_PREF + qid_orig

    query_lemmas, query_unlemm = nlp.procText(query)

    if query_lemmas == '':
        continue
    if query_lemmas in ignoreQueries:
        print(f"Ignoring query, which is found in specified query files. Raw query: '{query}' lemmatized query '{query_lemmas}'")

    query_toks = query_lemmas.split()
    if len(query_toks) >= minQueryTokQty:

        qrelList.append(QrelEntry(queryId=qid, docId=did, relGrade=1))

        # Entries are sorted by the query ID
        if prevQid != qid:
            doc = {DOCID_FIELD: qid,
                   TEXT_FIELD_NAME: query_lemmas,
                   TEXT_UNLEMM_FIELD_NAME: query_unlemm,
                   TEXT_RAW_FIELD_NAME: query.lower()}
            addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)

            docStr = json.dumps(doc) + '\n'
            outFileQueries.write(docStr)
            genQueryQty += 1
            if genQueryQty >= maxQueryQty:
                break

    prevQid = qid

    if ln % REPORT_QTY == 0:
        print('Processed %d input line' % ln)

print('Processed %d input lines' % ln)

writeQrels(qrelList, outFileQrelsName)

inpFile.close()
outFileQueries.close()

