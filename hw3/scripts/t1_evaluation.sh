python src/evaluation.py -o output/t1/remi_tf/t0.7p0.3 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl
python src/evaluation.py -o output/t1/remi_tf/t1.0p0.5 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl
python src/evaluation.py -o output/t1/remi_tf/t3.0p0.99 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl

python src/evaluation.py -o output/t1/remi_mb/t0.7p0.3 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl
python src/evaluation.py -o output/t1/remi_mb/t1.0p0.5 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl
python src/evaluation.py -o output/t1/remi_mb/t3.0p0.99 -d Pop1K7/representations/uncond/remi/ailab17k_from-scratch_remi/dictionary.pkl

python src/evaluation.py -o output/t1/cp_tf/t0.3p0.3 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl
python src/evaluation.py -o output/t1/cp_tf/t1.0p0.5 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl
python src/evaluation.py -o output/t1/cp_tf/t3.0p0.99 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl

python src/evaluation.py -o output/t1/cp_mb/t1.0p0.5 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl
python src/evaluation.py -o output/t1/cp_mb/t2.0p0.7 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl
python src/evaluation.py -o output/t1/cp_mb/t3.0p0.99 -d Pop1K7/representations/uncond/cp/ailab17k_from-scratch_cp/dictionary.pkl
