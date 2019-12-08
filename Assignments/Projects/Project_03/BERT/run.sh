python create_pretraining_data.py --input_file=./PRE_TRAIN_DIR/bert_*_pretrain.txt --output_file=./PRE_TRAIN_DIR/tf_examples.tfrecord --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --do_lower_case=True --max_seq_length=128 --max_predictions_per_seq=60 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5

python run_pretraining.py --input_file=./PRE_TRAIN_DIR/tf_examples.tfrecord --output_dir=./models/output --do_train=True --do_eval=True --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt --train_batch_size=4 --max_seq_length=128 --max_predictions_per_seq=60 --num_train_steps=100 --num_warmup_steps=10 --learning_rate=2e-5

python run_classifier_multi_labels_bert.py --task_name=multilabels --do_train=true --do_eval=true --data_dir=./data --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=./models/output/model.ckpt-100 --max_seq_length=128 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3 --output_dir=./checkpoint_bert

python run_multilabels_classifier.py --task_name=multilabels --do_train=true --do_eval=true --data_dir=./data --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=./models/output/model.ckpt-100 --max_seq_length=128 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3 --output_dir=./checkpoint_bert