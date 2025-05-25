# SMCE-Based Adversarial Example Detection

## Overview

Despite significant advancements in adversarial example detection, deep learning systems still face acute vulnerabilities to increasingly sophisticated perturbation strategies. This repository introduces a novel metric, Sliding Mask Confidence Entropy (SMCE), and a corresponding detection algorithm, Sliding Window Masking–Adversarial Example Detection (SWM-AED), which robustly identifies adversarial inputs by leveraging SMCE.

## Files

### `SWM-AED_smc_select_thresholds.py`

This script is designed to select optimal thresholds for different algorithms based on SMCE values. It processes the SMCE values obtained from both adversarial and clean samples to determine the most effective threshold that maximizes detection accuracy. The thresholds are crucial for distinguishing between adversarial and clean inputs.

### `SWM-AED_smc.py`

This script allows users to compute the SMCE values for both adversarial and clean samples. It provides a detailed analysis of the confidence entropy patterns, revealing how adversarial examples exhibit distinct entropy characteristics compared to clean samples. This script is essential for understanding the local confidence fluctuations captured by the sliding window mechanism.

## Usage

### Threshold Selection

To select thresholds for different algorithms, run the following command:

```bash
python SWM-AED_smc_select_thresholds.py
```

This script will analyze the SMCE values and output the optimal thresholds for each algorithm. These thresholds can then be used in the detection process to distinguish between adversarial and clean inputs.

### SMCE Analysis

To compute and analyze the SMCE values for adversarial and clean samples, run:

```bash
python SWM-AED_smc.py
```

This script will provide visualizations and numerical results, highlighting the differences in SMCE values between adversarial and clean samples. It helps in understanding the effectiveness of the SMCE metric in capturing local confidence fluctuations.

## Contributions

This repository presents a novel approach to adversarial example detection by leveraging SMCE. The SMCE metric captures local confidence fluctuations via a sliding window mechanism, revealing distinct entropy patterns in adversarial versus clean samples. The SWM-AED algorithm, built on this metric, demonstrates superior detection performance across diverse architectures and attack strategies. This work not only advances the theoretical understanding of adversarial vulnerabilities but also provides a practical and updatable defense mechanism against emerging threats in AI security.

## License


## Links to Datasets used in the paper
CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html (scroll down to second half of the page)
