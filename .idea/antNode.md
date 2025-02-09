Creating an object that reads the folder structure

collonyNode holds the object with all the antNodes.

The antNode is the recording node compose of the data and its metadata
data is store in parque structures and metadata as json

reads all the json files nested within the folders and create path pointers to the data

only when the data is call to load() the parque data is loaded as a mmap data. 

The user is allow to choose engine backend, in this case we will start with dask. future move to something in .rs

the node has a var call filters other call functions that is a list of functions that are apply to the array when excecute compute

each node is run independently but the collonyNode can add functions to all the nodes if needed 


visualization will be tricky, hopefully I can work on some sort of integration with rerun or build somethin based on egui

to compute the functions do: 

```py
filters = [
    (apply_notch_filter, {'frequency': notch_frequency, 'fs': fs, 'bandwidth': 1.0, 'order': 4}),
    (apply_bandpass_filter, {'lowcut': lowcut_bandpass, 'highcut': highcut_bandpass, 'fs': fs, 'order': 4}),
]


```