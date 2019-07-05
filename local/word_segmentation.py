#!/usr/bin/env python
# encoding=utf-8


import sys
import jieba
reload(sys)
sys.setdefaultencoding('utf-8')

if len(sys.argv) < 3:
  sys.stderr.write("word_segmentation.py <vocab> <trans> > <word-segmented-trans>\n")
  exit(1)

vocab_file=sys.argv[1]
trans_file=sys.argv[2]

jieba.set_dictionary(vocab_file)
for line in open(trans_file):
  #key,trans = line.strip().split('\t',1)
  res = line.strip().split('\t', 1)
  if len(res) == 2:
    key, trans = res
    words = jieba.cut(trans, HMM=False) # turn off new word discovery (HMM-based)
    new_line = key + '\t' + " ".join(words)
    print(new_line)
