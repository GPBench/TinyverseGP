for n in $(seq 1 10); do
        for i in $(seq 0 29); do
                sleep 1
                sbatch --time=1-00:00:00 --mem-per-cpu=3000 --job-name=tinyverse-max-cgp --wrap="python -m examples.analysis.max.test_max_plus_mul_cgp_single $d 1" --output="max-cgp-single-$d-$i.out"
        done
done
