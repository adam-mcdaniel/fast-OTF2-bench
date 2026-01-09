# fast-OTF2-bench

This project benchmarks `fastotf2`, a tool used to convert traces from OTF2 to CSV for quick loading into Python workflows.
Once converted to CSVs, Python notebooks can load them in seconds rather than minutes at a time, even for large traces.

## Project Files

|File|Description|
|-|-|
|`convert.py`|Performs conversion to CSV using OTF2's own Python API.|
|`otf2csv.c`|Performs conversion to CSV using the C library for OTF2.|
|`fastotf2-convert-and-load-bench.ipynb`|This notebook runs `fastotf2` on the traces and loads them into a structured API for representing exascale traces, managing `Ensemble`s of `Run`s which each contain `Node`s, the `Rank`s that run on them, and their metrics from the trace. This notebook benchmarks the speedup `fastotf2` provides over reading the data using the `otf2` module directly or using C to perform the conversion instead.|
|`read-and-convert-bench.sbatch`|This file runs `fastotf2`, `convert.py`, and `otf2csv` and logs their outputs and timings. Each converter logs its trace loading time, the time it takes to write the data back out, and the total conversion time, so the two APIs can be directly compared.|
|`frontier-1-node-single-HPL-run.tar.gz`|A trace from a single HPL run on a single node on the Frontier supercomputer at ORNL.|
|`frontier-16-node-single-HPL-run.tar.gz.part*`|A trace from a single HPL run on 16 nodes on the Frontier supercomputer at ORNL.|

## Build the Tools

First, clone the repo and its submodule.
```bash
$ git clone --recursive https://github.com/adam-mcdaniel/fast-OTF2-bench
```

#### Build the `fastotf2` Converter
Go to the fast-OTF2 directory and build with the makefile. Make sure you have [Chapel](https://chapel-lang.org/) installed!
```bash
$ cd fast-OTF2 && make
```

#### Build the C Converter
To build the C converter used for the results, install the [OTF2 bindings](https://scorepci.pages.jsc.fz-juelich.de/otf2-pipelines/doc.r4735/installationfile.html) and run the following:
```bash
$ gcc -O3 -o otf2csv ./otf2csv.c $(otf2-config --cflags) $(otf2-config --libs) $(otf2-config --ldflags)
```

## Run the Benchmarks

> [!WARNING]
> The scripts here all use an environment setup on the Frontier supercomputer. You will need to substitute the `LD_LIBRARY_PATH` configuration for your own machine -- it needs to contain the library files for your OTF2 installation.

#### Untar the Traces
Join all the trace parts together and untar them to get the trace files.
```bash
$ # Join all the trace parts
$ cat frontier-16-node-single-HPL-run.tar.gz.part_* > frontier-16-node-single-HPL-run.tar.gz
$ # Untar
$ tar -xvf frontier-1-node-single-HPL-run.tar.gz
$ tar -xvf frontier-16-node-single-HPL-run.tar.gz
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
This will run the `otf2csv` and `fastotf2` converters, and then load their data into a structured API that allows users to process the data in their traces and convert them into [Hatchet](https://hatchet.readthedocs.io/en/latest/) and [Thicket](https://thicket.readthedocs.io/en/latest/) profiles.