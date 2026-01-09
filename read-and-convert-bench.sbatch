#!/usr/bin/env bash
#SBATCH -A csc688
#SBATCH -N 1
#SBATCH -p extended
#SBATCH --exclusive
#SBATCH -t 06:00:00
#SBATCH -J chpl-otf2-convert-study
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -C nvme

# Set echo to log to a file
echo "Starting time conversion script at $(date)"

cd /lustre/orion/csc688/world-shared/scorep-amd/runs/fast-OTF2-bench/
echo "Current directory: $(pwd)"

source /lustre/orion/csc688/world-shared/chapel-2.7.0/util/setchplenv.bash

export LD_LIBRARY_PATH="/opt/rocm-6.4.1/lib:/lustre/orion/csc688/world-shared/scorep-amd/install/lib:$LD_LIBRARY_PATH"
export CHPL_RT_NUM_THREADS_PER_LOCALE=64
export METRICS="A2rocm_smi:::energy_count:device=0,A2rocm_smi:::energy_count:device=2,A2rocm_smi:::energy_count:device=4,A2rocm_smi:::energy_count:device=6,A2rocm_smi:::gpu_clk_freq_System:device=0:current,A2rocm_smi:::gpu_clk_freq_System:device=2:current,A2rocm_smi:::gpu_clk_freq_System:device=4:current,A2rocm_smi:::gpu_clk_freq_System:device=6:current,A2rocm_smi:::temp_current:device=0:sensor=1,A2rocm_smi:::temp_current:device=2:sensor=1,A2rocm_smi:::temp_current:device=4:sensor=1,A2rocm_smi:::temp_current:device=6:sensor=1,A2rocm_smi:::power_average:device=0:sensor=0,A2rocm_smi:::power_average:device=2:sensor=0,A2rocm_smi:::power_average:device=4:sensor=0,A2rocm_smi:::power_average:device=6:sensor=0,A2rocm_smi:::memory_busy_percent:device=0,A2rocm_smi:::memory_busy_percent:device=2,A2rocm_smi:::memory_busy_percent:device=4,A2rocm_smi:::memory_busy_percent:device=6,A2rocm_smi:::busy_percent:device=0,A2rocm_smi:::busy_percent:device=2,A2rocm_smi:::busy_percent:device=4,A2rocm_smi:::busy_percent:device=6,A2coretemp:::craypm:power,A2coretemp:::craypm:energy,A2coretemp:::craypm:freshness,A2coretemp:::craypm:cpu_energy,A2coretemp:::craypm:cpu_power,A2coretemp:::craypm:memory_energy,A2coretemp:::craypm:memory_power,A2coretemp:::craypm:accel0_energy,A2coretemp:::craypm:accel0_energy_timestamp,A2coretemp:::craypm:accel0_power,A2coretemp:::craypm:accel1_energy,A2coretemp:::craypm:accel1_energy_timestamp,A2coretemp:::craypm:accel1_power,A2coretemp:::craypm:accel2_energy,A2coretemp:::craypm:accel2_energy_timestamp,A2coretemp:::craypm:accel2_power,A2coretemp:::craypm:accel3_energy,A2coretemp:::craypm:accel3_energy_timestamp,A2coretemp:::craypm:accel3_power"

export CHPL_OTF2CSV_PATH=./fast-OTF2/trace_to_csv_parallel_real

function convert_trace {
    local trace_path=$1
    
    echo "Starting Chapel trace to CSV conversion for trace: $trace_path"
    START_TIME=$SECONDS
    /usr/bin/time -p $CHPL_OTF2CSV_PATH -nl1 --trace $trace_path --metrics $METRICS
    ELAPSED_TIME=$(($SECONDS - $START_TIME))
    echo "Chapel trace to CSV conversion completed in $ELAPSED_TIME seconds."
    echo "Cleaning up intermediate CSV files..."
    mkdir -p ./chapel_csv_outputs
    mv *.csv ./chapel_csv_outputs/
    rm ./*.csv

    echo "Starting Python trace to CSV conversion for trace: $trace_path"
    START_TIME=$SECONDS
    /usr/bin/time -p python3.11 ./convert.py $trace_path --metrics $METRICS --no-progress
    ELAPSED_TIME=$(($SECONDS - $START_TIME))
    echo "Python trace to CSV conversion completed in $ELAPSED_TIME seconds."
    echo "Cleaning up intermediate CSV files..."
    mkdir -p ./python_csv_outputs
    mv *.csv ./python_csv_outputs/
    rm ./*.csv
}

convert_trace "./frontier-1-node-single-HPL-run/traces.otf2"
echo "Single node conversions completed!"

convert_trace "./frontier-16-node-single-HPL-run/traces.otf2"
echo "Sixteen node conversions completed!"
