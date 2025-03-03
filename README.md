# dspAnt
This project takes inspiration from spikeinterface[^1] and branches only due to different requirements to handle emg data, 

to follow project progress follow this link https://www.tldraw.com/p/7m_PZy79ZkdbCX-wcoYRZ?d=v226.-37.1577.1157.EJte7OP_8H5jimpGy8Wyq

## Discord Channel 
https://discord.gg/jGPJTrSU

## Working prototype for electrophysiology data processing

### Core idea: https://www.tldraw.com/p/7m_PZy79ZkdbCX-wcoYRZ?d=v-188.522.2217.1628.page
- use dask for lazyloading data processing
- use pyarrow as data loading - aim for memory mapped
- use parquet as the standard file storage for optimized reading and storing
- graph nodes as the core phylosophy of the module - processing functions are attached to the raw data and processing can be access by requesting different node combinations
- metadata will be enforce:
  - When looking into other frameworks and how data is store by different products there are different design ideas behind the code. This make it hard to understand the data collected. The idea of this framework is to ensure that all dataset will be splitted into two major groups: Streams and Epocs. Streams will require not only having fs[sampling_frequency] but also units and other variables that describe the parquet file. This metadata will be name "**core**" in the metadata.json file, example:
```json
{
    "source": "StructType",
    "base": {
        "name": "RawG",
        "fs": 24414.0625,
        "number_of_samples": 35143680,
        "data_shape": [
            2,
            35143680
        ],
        "channel_numbers": 2,
        "channel_names": [
            "0",
            "1"
        ],
        "channel_types": [
            "float32",
            "float32"
        ],
        "channel_units":[
            "V",
            "V"
        ],
    },
    "other": {
        "code": 1199006034,
        "size": 2058,
        "type": 33025,
        "type_str": "streams",
        "ucf": "False",
        "dform": 0,
        "start_time": 0.0,
        "channel": [
            "1",
            "2"
        ],
        "save_path": "..\\data\\25-02-12_9882-1_testSubject_emgContusion\\drv_16-49-56_stim\\RawG.ant"
    }
}
```

### Current prototype: 
- 

### Prototype made with LLMs
Currently working on this project on my own, I use Claude, Chatgpt and MetaAI to reach to the current prototype. Errors and Bugs are expected use caususly 




