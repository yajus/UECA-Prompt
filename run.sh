#python ECPE.py \
#--test_only 0 \
#--checkpoint 0 \
#--checkpointpath 'checkpoint/ECPE/' \
#--savecheckpoint 0 \
#--save_path 'checkpoint/prompt_ECPE' \
#--dataset 'data_combine_ECPE_balance/' \
#--batch_size 8 \
#--learning_rate 1e-5 \
#--device '2'

#python ECE.py \
#--test_only True \
#--checkpoint True \
#--checkpointpath 'checkpoint/ECE/' \
#--savecheckpoint False \
#--save_path 'checkpoint/promt_ECE' \
#--dataset 'data_combine_ECE/' \
#--batch_size 8 \
#--learning_rate 1e-5 \
#--device '2'

#python CCRC.py \
#--test_only True \
#--checkpoint True \
#--checkpointpath 'checkpoint/CCRC/' \
#--savecheckpoint False \
#--save_path 'checkpoint/prompt_CCRC' \
#--dataset 'data_combine_CCRC/' \
#--batch_size 8 \
#--learning_rate 1e-5 \
#--device '2'

#python ECPE_M2M.py \
#--test_only True \
#--checkpoint True \
#--checkpointpath 'checkpoint/ECPE_M2M/' \
#--savecheckpoint False \
#--save_path 'checkpoint/prompt_ECPC_M2M' \
#--dataset 'data_combine_ECPE/' \
#--batch_size 8 \
#--learning_rate 1e-5\
#--device '2'