#!/bin/bash

for g in 5 10
do 
    for nle in 4, 15
    do
        for lr_main in 0.005 0.001 0.0005 0.0001
        do
            for seed_init in 1 2 3
            do
                for sigma_v1 in 0.1 0.5 1.0 2.0
                do 
                    for sigma_Win in 0.1 0.5 1.0 2.0
                    do
                        for prediction_offset in 1 50 100
                        do  
                            jobfile="sgf_job"
                            echo "Submitting job with g: $g nle: $nle lr_main $lr_main seed_init $seed_init sigma_v1 $sigma_v1 sigma_Win $sigma_Win offset $prediction_offset"
                            
                            echo "#!/bin/sh" >> $jobfile
                            echo "#SBATCH --job-name=d3" >> $jobfile
                            echo "#SBATCH -c 1" >> $jobfile
                            echo "#SBATCH --time=24:00:00" >> $jobfile
                            echo "#SBATCH --mem-per-cpu=2gb" >> $jobfile
                            echo "#SBATCH --cpus-per-task=2" >>$jobfile
                            echo "python lyapunov_snn_sg/train.py --output_dir=/engram/naplab/users/ms7240/results 
                            --g=$g --nle=$nle --lr_main=$lr_main --seed_init=$seed_init 
                            --sigma_v1=$sigma_v1 --sigma_Win=$sigma_Win --prediction_offset=$prediction_offset" >> $jobfile
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
