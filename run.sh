CUDA_VISIBLE_DEVICES=1 python run_drcd.py \
  --do_train=False \
  --do_predict=True \
  --train_batch_size=6 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --do_lower_case=True \
  --max_seq_length=512 \
  --doc_stride=128 \
  # --vocab_file=$BERT_MODEL_DIR/vocab.txt \
  # --bert_config_file=$BERT_MODEL_DIR/bert_config.json \
  # --init_checkpoint=$BERT_MODEL_DIR/bert_model.ckpt \
  # --train_file=$DRCD_DIR/DRCD_training.json \
  # --predict_file=$DRCD_DIR/DRCD_test.json \
  # --output_dir=$OUTPUT_DIR/