for n in $(seq 1 30); do
        for i in $(seq 0 29); do
                sleep 1
                sbatch --time=1-00:00:00 --mem-per-cpu=3000 --job-name=tinyverse-and-cgp-log --wrap="python -m examples.analysis.max.test_max_plus_mul_cgp_single_log $d 2" --output="max-cgp-single-log-$d-$i.out"
        done
done
