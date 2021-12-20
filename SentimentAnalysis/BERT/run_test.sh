python run_classifier.py \
  --task_name=emotion \
  --do_predict=true \
  --data_dir=glue_data/PGGFX \
  --vocab_file=models/cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=models/cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=emotion_output \
  --max_seq_length=128 \
  --output_dir=./tmp/mrpc_output/