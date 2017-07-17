#!/bin/bash

python3 ../source/faro_datamaker.py --working_dir=../test --epsilon=1.0 --coarse_timestep=0.1 --outFileStem=faro --meas_spacing=12 --assim_cycle_length=10
python3 ../source/faro.py --working_dir=../test --epsilon=1.0 --coarse_timestep=0.1 --outFileStem=faro --assim_cycle_length=10

