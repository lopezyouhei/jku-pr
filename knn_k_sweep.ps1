$start = 1
$end = 9
$step = 1

for ($n_neighbors = $start; $n_neighbors -le $end; $n_neighbors += $step) {
    python .\PR\metrics\knn_main.py --n_neighbors $n_neighbors
}