time_lim=100

python -O exp_runner.py linear --method ghost     --timeout $time_lim
python -O exp_runner.py linear --method ghost     --timeout $time_lim --epsilon 0.5
python -O exp_runner.py linear --method ghost-ecg --timeout $time_lim
python -O exp_runner.py linear --method greedy    --timeout $time_lim

mkdir -p $PWD/../data/exp/linear-micp

for num_sets in $(seq 5 25)
do
    for seed in $(seq 0 11)
    do
        echo "Running MICP with seed $seed and num_sets $num_sets"
        timeout $time_lim python micp_runner.py linear --seed $seed --num_sets $num_sets > $PWD/../data/exp/linear-micp/$seed-$num_sets.log
    done
done

python $PWD/../data/extract_micp_res.py linear
