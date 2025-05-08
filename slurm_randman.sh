#!/bin/bash

declare -a g_list
declare -a nle_list
declare -a lr_main_list
declare -a seed_init_list
declare -a sigma_v1_list
declare -a sigma_Win_list
declare -a prediction_offset_list

g_list = (5)
nle_list = (4)
lr_main_list = (0.005)
seed_init_list = (1)
sigma_v1_list = (5)
sigma_Win_list = (5)
prediction_offset_list = (50, 100)


for g in "${g_list[@]}"; do 
    for nle in "${nle_list[@]}"; do
        for lr_main in "${lr_main_list[@]}"; do
            for seed_init in "${seed_init_list[@]}"; do
                for sigma_v1 in "${sigma_v1_list[@]}"; do 
                    for sigma_Win in"${sigma_Win_list[@]}"; do
                        for prediction_offset in "${prediction_offset_list[@]}"; do  
                            jobfile="sgf_job"
                            echo "Submitting job with g: $g nle: $nle lr_main $lr_main seed_init $seed_init sigma_v1 $sigma_v1 sigma_Win $sigma_Win offset $prediction_offset"
                            
                            echo "#!/bin/sh" >> $jobfile
                            echo "#SBATCH --job-name=d3" >> $jobfile
                            echo "#SBATCH -c 1" >> $jobfile
                            echo "#SBATCH --time=24:00:00" >> $jobfile
                            echo "#SBATCH --mem-per-cpu=2gb" >> $jobfile
                            echo "#SBATCH --cpus-per-task=1" >>$jobfile
                            echo "python lyapunov_snn_sg/train.py --output_dir=/engram/naplab/users/ms7240/results --g=$g --nle=$nle --lr_main=$lr_main --seed_init=$seed_init --sigma_v1=$sigma_v1 --sigma_Win=$sigma_Win --prediction_offset=$prediction_offset" >> $jobfile
                            echo "date" >> $jobfile
                            
                            sbatch $jobfile
                            echo "waiting"
                        done
                    done
                done
            done
        done
    done
done

wait
