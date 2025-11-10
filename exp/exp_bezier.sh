time_lim=100

python -O exp_runner.py bezier --method ghost     --timeout $time_lim
python -O exp_runner.py bezier --method ghost     --timeout $time_lim --epsilon 0.5
python -O exp_runner.py bezier --method ghost-ecg --timeout $time_lim
python -O exp_runner.py bezier --method greedy    --timeout $time_lim

mkdir -p $PWD/../data/exp/bezier-micp

for num_sets in $(seq 5 25)
do
    for seed in $(seq 0 11)
    do
        echo "Running MICP with seed $seed and num_sets $num_sets"
        timeout $time_lim python micp_runner.py bezier --seed $seed --num_sets $num_sets > $PWD/../data/exp/bezier-micp/$seed-$num_sets.log
    done
done

python $PWD/../data/extract_micp_res.py bezier
