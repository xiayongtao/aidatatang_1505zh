#!/bin/bash

. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "train_lms.sh <lexicon> <word-segmented-text> <dir>"
  echo " e.g train_lms.sh data/local/dict/lexicon.txt data/local/train/text data/local/lm"
  exit 1;
fi

lexicon=$1
text=$2
dir=$3

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

kaldi_lm=`which train_lm.sh`
#echo "{$kaldi_lm}--------------------------------------"
if [ -z $kaldi_lm ]; then
  echo "$0: train_lm.sh is not found. That might mean it's not installed"
  echo "$0: or it is not added to PATH"
  echo "$0: Use the script tools/extras/install_kaldi_lm.sh to install it"
  exit 1
fi

mkdir -p $dir
cleantext=$dir/text.no_oov

cat $text | awk -v lex=$lexicon 'BEGIN{while((getline<lex) >0){ seen[$1]=1; } }
  {for(n=1; n<=NF;n++) {  if (seen[$n]) { printf("%s ", $n); } else {printf("<UNK> ");} } printf("\n");}' \
  > $cleantext || exit 1;

cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | sort | uniq -c | \
   sort -nr > $dir/word.counts || exit 1;

# Get counts from acoustic training transcripts, and add  one-count
# for each word in the lexicon (but not silence, we don't want it
# in the LM-- we'll add it optionally later).
cat $cleantext | awk '{for(n=2;n<=NF;n++) print $n; }' | \
  cat - <(grep -w -v '!SIL' $lexicon | awk '{print $1}') | \
   sort | uniq -c | sort -nr > $dir/unigram.counts || exit 1;

# note: we probably won't really make use of <UNK> as there aren't any OOVs
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<UNK>" > $dir/word_map \
   || exit 1;

# note: ignore 1st field of train.txt, it's the utterance-id.
cat $cleantext | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=2;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz \
   || exit 1;

train_lm.sh --arpa --lmtype 3gram-mincount $dir || exit 1;

# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz

echo "local/train_lms.sh succeeded"
exit 0
