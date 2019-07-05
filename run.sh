#!/bin/bash

trn_set=/home/ldf/corpus/aidatatang_1505zh/corpus/train
dev_set=/home/ldf/corpus/aidatatang_1505zh/corpus/dev
tst_set=/home/ldf/corpus/aidatatang_1505zh/corpus/test

nj=4
stage=0
gmm_stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage
fi

# chain
if [ $stage -le 3 ]; then
  local/chain/run_tdnn_1a.sh --nj $nj
fi

local/show_results.sh

exit 0;
