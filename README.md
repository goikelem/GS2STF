<div align="center">

# 🌊 S2S Teleconnection Forecasting

**A robust, modular climate data pipeline and forecasting toolkit restructured for scalability and precision.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Refactored](https://img.shields.io/badge/status-restructured-success.svg)](#)

</div>

---

> **⚠️ Restructuring Notice:** This repository has been cleanly reorganized from the flat `SST_Climate_Indices` project into a modular, specification-compliant architecture. **No core algorithms were changed.** The SST-index analysis, SPI/SPEI workflow, and SST-rainfall teleconnections behave identically to the original project.

---

## 🚀 Quick Start

Get up and running with dependencies and CLI commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Display help menu
python main.py --help

# 1. SST index forecasting (no NetCDF dependency required)
python main.py --from-regional-csv monthly_regional_sst.csv --fast

# 2. SPI / SPEI drought indices and spatial analysis
python main.py --index both --classification --spatial-analysis --make-maps

# 3. SST-rainfall teleconnections
python main.py --teleconnection --season JJAS

# Run test suites
python tests/test_kernels.py && python tests/test_tele_kernels.py && python tests/test_station.py && python tests/test_seasonal.py && python tests/test_ensemble.py
