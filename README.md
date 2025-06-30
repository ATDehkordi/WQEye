# WQEye

[![Tutorial](https://img.shields.io/badge/Step--by--Step%20Tutorial-PDF%20Guide-blue)](docs/Step-by-Step github.pdf)
[![Installation](https://img.shields.io/badge/Installation-PDF%20Guide-green)](docs/Installation.pdf)

## 1. System Requirements and Compatibility

WQEye is developed entirely in **Python**, making it cross-platform and compatible with major operating systems, including **Windows**, **macOS**, and **Linux distributions** such as Ubuntu. 
The software includes six modules. All modules, except RS Sampling, run locally on the user's computer, while RS Sampling relies on Google Earth Engine for processing. Machine learning models implemented in WQEye run on the user's computer **CPU by default**. This was done to make the software accessible to a broad range of users. However, if a **CUDA-capable GPU** is available on the system, the software will automatically use it to accelerate processing for ANNs and KANs. So, WQEye is not dependent on hardware-specific features such as GPUs and is designed to function efficiently on both basic and advanced computer configurations.

> **Note:** An active internet connection is required to access the **Google Earth Engine (GEE)** in the **RS Sampling Module**.

### Minimum Recommended Requirements

| Component       | Specification            |
|-----------------|--------------------------|
| Processor       | Dual-core CPU or higher   |
| Memory          | At least 4 GB RAM         |
| Operating System| No difference             |
| Python Version  | Python 3.12.x             |
| Internet        | Required for RS Sampling via GEE |

---

The developers successfully installed and tested WQEye on multiple platforms, ranging from high-performance workstations to standard consumer laptops. The following table summarizes the system specifications used for testing:

### Tested System Specifications

| Operating System                       | Processor                            | Memory (RAM) | Graphics (GPU)                          |
|----------------------------------------|--------------------------------------|---------------|------------------------------------------|
| Ubuntu 22.04.4 LTS                     | 13th Gen Intel® Core™ i9-13900       | 62 GB         | NVIDIA® GeForce® RTX 3050 (Dedicated GPU) |
| macOS Sonoma 14.6 (MacBook Air 2024)   | Apple® M3 Chip (8-Core CPU)          | 24 GB         | Apple® M3 Integrated GPU (10-Core GPU)  |
| Windows 10 Pro 22H2 (64-bit)           | Intel® Core™ i7-6700HQ               | 16 GB         | NVIDIA® GeForce® GTX 960M (8 GB Memory) |

