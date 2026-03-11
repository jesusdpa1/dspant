# dspAnt 🐜

A flexible and powerful framework for processing time-series data, with a focus on neural and physiological signals.

## 🌟 Project Vision

dspAnt aims to simplify complex signal processing workflows by providing an intuitive, performant, and scalable approach to data analysis.

## 🚀 Key Design Principles

### Data Processing Philosophy

```mermaid
flowchart TD
    A[Raw Data] --> B{Dask Lazy Loading}
    B --> C[Memory-Efficient Processing]
    C --> D[Parquet Storage]
    D --> E[Flexible Node-Based Analysis]
    
    subgraph Data Structure
        F[Streams] --> G[Continuous Signals]
        H[Epochs] --> I[Event-Based Data]
    end
```

### Why Our Approach?

1. **Lazy Loading**: Process large datasets without memory constraints
2. **Standardized Metadata**: Consistent data description across different sources
3. **Flexible Processing**: Attach processing functions to raw data dynamically
4. **Scalable Architecture**: Designed for complex scientific workflows

### Metadata Example

We use a structured metadata approach to provide clear, comprehensive information about your data:

```json
{
    "base": {
        "name": "RawEMG",
        "sampling_rate": 24414.0625,
        "total_samples": 35143680,
        "channels": {
            "count": 2,
            "names": ["Channel1", "Channel2"],
            "units": ["V", "V"]
        }
    },
    "recording_details": {
        "date": "2024-02-25",
        "subject_id": "9882-1",
        "experiment_type": "EMG Contusion"
    }
}
```

## 🛠 Current Features

- Dask-powered lazy data processing
- PyArrow-based memory-mapped reading
- Parquet file storage
- Flexible node-based processing
- Support for streams and epoch-based data

## 📡 Community & Support

- **Discord**: [Join our Community](https://discord.gg/jGPJTrSU)
- **Inspiration**: [SpikeInterface](https://github.com/SpikeInterface/spikeinterface.git)

## 🧠 Development Approach

This project is developed with the assistance of AI language models (Claude, ChatGPT, Meta AI). 

**Disclaimer**: As an evolving prototype, expect ongoing improvements and potential bugs.

## 🗺️ Project Roadmap

```mermaid
timeline
    title dspAnt Development Roadmap
    section Prototype Foundation
        1 : Initial Architecture Design
        : Basic Stream/Epoch Node Implementation
        : Core Processing Utilities
    
    section Core Functionality
        2 : Robust Signal Processing Modules
        : Comprehensive Metadata Handling
        : Performance Optimization
        : Documentation Expansion
    
    section Specialized Modules
        EMG Processing Module : Low-level Signal Analysis
            : Onset Detection
            : Fatigue Analysis
        High-Density EMG Processing
            : Advanced Spatial Analysis
            : Multi-Channel Processing Techniques
        Machine Learning Integration
    
    section Community & Scaling
        3 : Open-Source Release
        : Community Feedback Collection
        : Bug Fixes and Stability Improvements
        : Compatibility Enhancements
        : Potential First Stable Release

```

## 🤝 Contributions

Interested in contributing? We welcome:
- Bug reports
- Feature suggestions
- Code contributions
- Documentation improvements

## 📊 Technology Stack

- **Language**: Python
- **Lazy Processing**: Dask
- **Data Storage**: PyArrow, Parquet
- **Signal Processing**: NumPy, SciPy
- **Performance**: Numba

## 📦 Installation (Coming Soon)

### Current 
``` bash
git clone repo
# cd to directory
# --extra for stft functionality

uv sync --extra cu124
```
### Coming Soon
```bash
pip install dspant
```

## 📚 Quick Example
<img src=".figures\img\emg_onset_detection_absolute.png" alt="Project Architecture" width="600" />

## License

[To be determined]

## References

[1] SpikeInterface: https://github.com/SpikeInterface/spikeinterface.git


## License and Disclaimer

### Liability Disclaimer

**IMPORTANT: USE AT YOUR OWN RISK**

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS SOFTWARE.

Users are solely responsible for:
- Verifying the accuracy and appropriateness of results
- Ensuring proper implementation in their specific use case
- Checking and validating all outputs
- Any consequences resulting from the use of this software
