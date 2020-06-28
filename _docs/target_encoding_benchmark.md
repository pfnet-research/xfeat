# Target Encoding with xfeat

![xfeat_target_encoding_image](./target_encoding_image.png)

Target encoding can be greatly accelerated by using GPUs. When using GPUs, `xfeat.TargetEncoder` accesses cuDF and CuPy internally.

## Benchmark

For the detailed experiment scripts and output logs, please refer to [/examples/benchmark](#).

## Data

* record size: 1M ~ 10M records.
* cardinality: 5,000
* num folds: 5

## Environment

We ran all experiments on a single Linux server (AWS p3.2xlarge) with the following specifications:

* p3.2xlarge on AWS EC2.
* OS: Ubuntu 18.04
* Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
* GPU: NVIDIA Tesla V100 x1
* Docker image: `smly/xfeat-cudf` ([Dockerfile](../examples/benchmark/docker/Dockerfile))

## Results

![xfeat_target_encoding_image](./benchmark_target_encoding.png)

## Usage of benchmark script

```bash
$ docker run -it --rm \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -u $(id -u $USER):$(id -g $USER) \
    -v /home/ubuntu/xfeat:/root \
    smly/xfeat-cudf bash

ubuntu$ python examples/benchmark/target_encoder.py
```
