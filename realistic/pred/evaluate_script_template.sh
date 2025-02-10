model=Your-Model-Name 

contextlengths=(zero_context 8k 16k 32k) 
for contextlength in "${contextlengths[@]}" 
do 
    modes=(2 3) # using medium and hard modes 
    for mode in "${modes[@]}" 
    do 
        op=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30) # adjust based on your needs 
        for i in "${op[@]}" 
        do 
            python evaluate_model_template.py --op $i --ip 20 --add_fewshot --limit 100 --testsuite $contextlength --force --modelname $model --batch_size 1 --d $mode --local_rank 0 | tee -a your-output-file.txt 
        done 
    done 
done 
