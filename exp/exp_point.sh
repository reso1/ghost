time_lim=100

python -O exp_runner.py point --method ghost     --timeout $time_lim 
python -O exp_runner.py point --method ghost     --timeout $time_lim --epsilon 0.5
python -O exp_runner.py point --method ghost-ecg --timeout $time_lim
python -O exp_runner.py point --method greedy    --timeout $time_lim

mkdir -p $PWD/../data/exp/point-micp

for seed in $(seq 0 11)
do
    for num_sets in $(seq 5 25)
    do
        echo "Running MICP with seed $seed and num_sets $num_sets"
        timeout $time_lim python micp_runner.py point --seed $seed --num_sets $num_sets > $PWD/../data/exp/point-micp/$seed-$num_sets.log
    done
done

python $PWD/../data/extract_micp_res.py point
