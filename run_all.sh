#!/usr/bin/env bash
# Entry point for the Docker image.  Usage: run_all.sh [all|fig3|extension|validate|--flags...]
set -euo pipefail
cd /nngp
OUT=/nngp/output
mkdir -p "$OUT"

# Paper Figure 3 hyperparameters (caption): depth=3, sigma_w^2=2.0, sigma_b^2=0.2
HP='depth=3,weight_var=2.0,bias_var=0.2'
NUM_TRAIN="${NUM_TRAIN:-5000}"     # paper: 50k (see README "Deviations")
NUM_EVAL="${NUM_EVAL:-10000}"      # full MNIST test set -> 100 bins of 100
TRAIN_SIZES="${TRAIN_SIZES:-250,500,1000,2000}"  # + NUM_TRAIN row reused from fig3

fig3() {
  rm -f "$OUT/fig3_metrics.csv"
  echo "=== [1/2] Figure 3 reproduction: MNIST, n_train=$NUM_TRAIN, tanh+ReLU (paper) and +GELU (extension) ==="
  python uncertainty_plot.py --dataset=mnist --num_train="$NUM_TRAIN" \
      --num_eval="$NUM_EVAL" --hparams="$HP" --nonlinearities=tanh,relu,gelu \
      --paper_output_file="$OUT/fig3_mnist_tanh_relu.png" \
      --output_file="$OUT/fig3_mnist_tanh_relu_gelu.png" \
      --metrics_file="$OUT/fig3_metrics.csv"
}

extension() {
  echo "=== [2/2] Extension: GELU vs tanh/ReLU across training-set sizes ($TRAIN_SIZES) ==="
  python extension_gelu.py --hparams="$HP" --train_sizes="$TRAIN_SIZES" \
      --num_eval="$NUM_EVAL" --output_dir="$OUT" \
      --extra_metrics_csv="$OUT/fig3_metrics.csv"
}

mode="${1:-all}"
case "$mode" in
  all)       fig3; extension ;;
  fig3)      fig3 ;;
  extension) extension ;;
  validate)  python make_grid.py --validate ;;
  --*)       exec python uncertainty_plot.py "$@" ;;
  *)         exec "$@" ;;
esac

echo
echo "================ SUMMARY ================"
for f in "$OUT/fig3_metrics.csv" "$OUT/ext_metrics.csv"; do
  [ -f "$f" ] && python -c "import csv,sys;[print('  '.join('%-13s'%c for c in r)) for r in csv.reader(open(sys.argv[1]))]" "$f" && echo
done
echo
echo "Outputs written inside the container to $OUT:"
ls -1 "$OUT"
echo "Tip: add  -v \"\$(pwd)/output\":/nngp/output  to 'docker run' to copy them to your machine."
