# fast-OTF2-bench

This project benchmarks `fastotf2`, a tool used to convert traces from OTF2 to CSV for quick loading into Python workflows.
Once converted to CSVs, Python notebooks can load them in seconds rather than minutes at a time, even for large traces.

## Project Files

|File|Description|
|-|-|
|`convert.py`|Performs conversion to CSV using OTF2's own Python API. It takes advantage of the same parallelism of the `fastotf2` version.|
|`fastotf2-convert-and-load-bench.ipynb`|This notebook runs `fastotf2` on the traces and loads them into a structured API for representing exascale traces, managing `Ensemble`s of `Run`s which each contain `Node`s, the `Rank`s that run on them, and their metrics from the trace. This notebook benchmarks the speedup `fastotf2` provides over reading the data using the `otf2` module directly.|
|`read-and-convert-bench.sbatch`|This file runs `fastotf2` and `convert.py` and logs their outputs. Each converter logs its trace loading time, the time it takes to write the data back out, and the total conversion time, so the two APIs can be directly compared.|
|`frontier-1-node-single-HPL-run/`|A trace from a single HPL run on a single node on the Frontier supercomputer at ORNL.|
|`frontier-16-node-single-HPL-run/`|A trace from a single HPL run on 16 nodes on the Frontier supercomputer at ORNL.|

## Run the Benchmarks

First, clone the repo and its submodule.
```bash
$ git clone --recursive https://github.com/adam-mcdaniel/fast-OTF2-bench
```

Then, you can run the following benchmarks:
#### Read and Convert

```bash
$ # Bench the reading/conversion speeds
$ sbatch read-and-convert-bench.sbatch
```

The outputs will be in SLURM logs: `slurm-%j.out` and `slurm-%j.err`.

#### Convert and Load into Workflow

Open and run the `fastotf2-convert-and-load-bench.ipynb` notebook!

<!--
C read time (16 nodes): 766.25
C read time (1 node): 7.22
-->