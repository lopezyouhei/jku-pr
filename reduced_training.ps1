$start = 0.9
$end = 0.1
$step = -0.1

for ($reduce_factor = $start; $reduce_factor -ge $end; $reduce_factor += $step) {
    python .\PR\train.py --reduce_factor $reduce_factor
}