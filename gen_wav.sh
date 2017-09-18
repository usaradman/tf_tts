#!/bin/bash
#  gen_wav.sh Author "Jinba Xiao <usar@npu-aslp.org>" Date 08.09.2017

cmp_dir=$1
lf0_dir=$cmp_dir/lf0
mgc_dir=$cmp_dir/mgc
wav_dir=$cmp_dir/wav
LSP_ORDER=41

python ./gen_param.py $cmp_dir $lf0_dir $mgc_dir $LSP_ORDER 0
[ -e $wav_dir  ] && rm -rf $wav_dir
mkdir $wav_dir
for file in $(ls $lf0_dir); do
    echo "synthesize $(basename $file .lf0).wav"; \
    #/export/expts/jbxiao/workspace/code/speech/tts/bazel-bin/engine/lpc_vocoder/synthesizer_main --sampling_rate=16000 --frame_period=80 --lsp_file=$mgc_dir/$(basename $file .lf0).mgc --lf0_file=$lf0_dir/$file --wave_save_file=$wav_dir/$(basename $file .lf0).wav
    ./bd_vocoder $mgc_dir/$(basename $file .lf0).mgc $lf0_dir/$file $wav_dir
done
