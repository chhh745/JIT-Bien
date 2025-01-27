# JIT-Bien
# JIT-Bien Implementation
All of our experiments were performed on an NVIDIA A800, and it is recommended to use the same graphics card or reduce the train_batch_size To train JIT-Bien, run the following command:
## Train and evaluate our JIT-Bien and JIT-Fine in the JIT-DP task.
···python
python -m JITBien.concat.run --output_dir=model/jitbien/saved_models_concat/checkpoints --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=microsoft/codebert-base --do_train --train_data_file data/jitfine/changes_train.pkl data/jitfine/features_train.pkl --eval_data_file data/jitfine/changes_valid.pkl data/jitfine/features_valid.pkl --test_data_file data/jitfine/changes_test.pkl data/jitfine/features_test.pkl --epoch 50 --max_seq_length 512  --max_msg_length 64 --train_batch_size 24 --eval_batch_size 128 --learning_rate 1e-5  --max_grad_norm 1.0  --evaluate_during_training --feature_size 14 --seed 42 --patience 10 --max_codeline_length 256 --max_codeline_token_length 64 --buggy_lines_file data/jitbien/train_buggy_commit_lines_df.pkl data/jitbien/valid_buggy_commit_lines_df.pkl data/jitbien/test_buggy_commit_lines_df.pkl --dp_loss_weight 0.3 --dl_loss_weight 0.7 2>&1| tee model/jitbien/saved_models_concat/train.log···
'''
## Train and evaluate our JIT-Bien and JIT-Fine in the JIT-DL task.
sh train_jitbien.sh
