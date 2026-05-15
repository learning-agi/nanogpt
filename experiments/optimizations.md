# Optimization Experiments

- GPT2 architecture
- B = 16
- T = 1024
- 1 x A100 40GB GPU (PCIe)

| Data Type                    | Additional Optimization | Average Time (per step)                        | TPS        | Memory Usage |
|:----------------------------:|:-----------------------:|:----------------------------------------------:|:----------:|:------------:|
| FP32                         |                         | 1118.39ms                                      | 14,730.81  | 35.6GB       |
| TF32                         |                         | 425.06ms                                       | 40,038.76  | 35.6GB       |
| BF16                         |                         | 367.69ms                                       | 47,278.59  | 33.9GB       |
| BF16                         | Compile                 | 263.20ms (compilation time > 30 secs)          | 105,520.53 | 23.8GB       |
| BF16                         | Prev + FlashAttention   | 197.49ms                                       | 143,213.67 | 14.7GB       |
| BF16                         | Prev + All powers of 2  | 194.74ms                                       | 147,669.85 | 13.8GB       |
| BF16                         | Prev + DDP              | 2438.89m                                       | 291,865.23 | 13.8GB       |