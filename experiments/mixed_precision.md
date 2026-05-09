# Mixed Precision Experiments

- GPT2 architecture
- B = 16
- T = 1024
- 1 x A100 40GB GPU (PCIE)

|Data Type|Average Time (per step)|TPS|Memory Usage|
|:---------:|:-----------------------:|:------:|:----:|
| FP32    | 1118.39ms             | 14,730.81| 35.6GB |
| TF32.   | 425.06ms              | 40,038.76| 35.6GB |
| BF16    | 367.69ms              | 47,278.59| 33.9GB |
| BF16 (with torch.compile)    | 263.20ms (1st step compilation time > 30 secs)              | 105,520.53| 23.8GB |