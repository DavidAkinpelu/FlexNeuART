#!/usr/bin/env python
# Convert MSMARCO queries
import sys
import json
import argparse
import pytorch_pretrained_bert

sys.path.append('.')

from scripts.data_convert.text_proc import SpacyTextParser
from scripts.data_convert.convert_common import STOPWORD_FILE, BERT_TOK_OPT_HELP, BERT_TOK_OPT, \
    FileWrapper, readStopWords, addRetokenizedField
from scripts.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, BERT_BASE_MODEL, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, \
    IMAP_PROC_CHUNK_QTY, REPORT_QTY, SPACY_MODEL

parser = argparse.ArgumentParser(description='Convert MSMARCO-adhoc queries.')
parser.add_argument('--input', metavar='input file', help='input file',
                    type=str, required=True)
parser.add_argument('--output', metavar='output file', help='output file',
                    type=str, required=True)
parser.add_argument('--min_query_token_qty', type=int, default=0,
                    metavar='min # of query tokens', help='ignore queries that have smaller # of tokens')
parser.add_argument('--' + BERT_TOK_OPT, action='store_true', help=BERT_TOK_OPT_HELP)

args = parser.parse_args()
print(args)
arg_vars = vars(args)

inpFile = FileWrapper(args.input)
outFile = FileWrapper(args.output, 'w')
minQueryTokQty = args.min_query_token_qty

stopWords = readStopWords(STOPWORD_FILE, lowerCase=True)
print(stopWords)
nlp = SpacyTextParser(SPACY_MODEL, stopWords, keepOnlyAlphaNum=True, lowerCase=True)

if BERT_TOK_OPT in arg_vars:
    print('BERT-tokenizing input into the field: ' + TEXT_BERT_TOKENIZED_NAME)
    bertTokenizer = pytorch_pretrained_bert.BertTokenizer.from_pretrained(BERT_BASE_MODEL)

# Input file is a TSV file
ln = 0
for line in inpFile:
    ln += 1
    line = line.strip()
    if not line:
        continue
    fields = line.split('\t')
    if len(fields) != 2:
        print('Misformated line %d ignoring:' % ln)
        print(line.replace('\t', '<field delimiter>'))
        continue

    did, query = fields

    query_lemmas, query_unlemm = nlp.procText(query)

    query_toks = query_lemmas.split()
    if len(query_toks) >= minQueryTokQty:
        doc = {DOCID_FIELD: did,
               TEXT_FIELD_NAME: query_lemmas,
               TEXT_UNLEMM_FIELD_NAME: query_unlemm,
               TEXT_RAW_FIELD_NAME: query.lower()}
        addRetokenizedField(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bertTokenizer)

        docStr = json.dumps(doc) + '\n'
        outFile.write(docStr)

    if ln % REPORT_QTY == 0:
        print('Processed %d queries' % ln)

print('Processed %d queries' % ln)

inpFile.close()
outFile.close()
