import subprocess
import json
from pathlib import Path
from helper_functions import loaddata

import os
import glob

def get_ODB_results(odb_name,instanceName,field_outputs,stepName,parameters,save_path = None,abaqus_version:str = 'abaqus',delete_json = True,return_mat = True,*,debug = False):
    '''
    Extract results from Abaqus ODB files into a .mat format. 
    NOTE BECAUSE OF HOW .MAT DICTIONARIES WORK PLEASE USE THE BUILT IN `loaddata` function to load from .mat files as this will help clean up trailing and leading whitespaces

    This function requires a subprocess call as the Abaqus CAE python kernel must be invoked to manipulate ODB files. The data manipulation is performed by the object `ODB_Exporter`
    found in `get_results_Abaqus_API.py`

    Inputs:
    - odb_name: str, a str representing the path to an odb file e.g. 'TubeCrush.odb'
    - instanceName: str, Name of instance to extract results from e.g. 'TUBE-1'
    - field_outputs: dict, dict of outputs to extract from. The key is the primary variable and the values is a tuple of invariants of components e.g. `{'U':('U1','U2','U3')}`
    - stepName: name of step to extract results from e.g. 'Step1'
    - parameters: dict, dictionary of global parameters that define the instance keys are the names and values are the value e.g. `{'steel':50.}`
    - save_path: str output path of the .mat file e.g 'test.mat'
    - abaqus_version: str abaqus version to use by default 'abaqus' will call the latest version. Must be Abaqus 2024 or later. For a specific version use the string: `abq[version]` e.g `abq2025` or `abq2024hf3` if a hotfix is applied
    - delete_json: bool, whether to delete the intermediate json files for parameters and dictionaries default is True
    - return__mat: bool, whether to return the saved .mat file for further usage default is True. If False returns None after successful extraction
    - debug: bool returns the subprocess object. Useful if an error has occurred default is False. This overides return_mat

    ```python
        from torch_DE.discrete.Abaqus import get_ODB_results

        x = get_ODB_results(
        instanceName = 'TUBE-1',
        field_outputs = {'U':('U1','U2','U3')},
        stepName = 'TubeCrush',
        parameters = {'p':50.},
        save_path = 'test.mat',
        odb_name = 'TubeCrush.odb',
        abaqus_version = 'abq2024hf3', 
        )
    ```
    will extract the displacement (U1,U2 and U3) from the instance 'TUBE-1' from TubeCrush.odb

    Outputs:
    - `save_path.mat` a Matlab .mat dictionary of arrays file. Because of how strings are implemented in matlab arrays (all strings are same size) use `loaddata` from the torch_DE abaqus Module instead of `scipy.loadmat`
    The form is set to compact mode. The dictionary has the following contents:
        - `part name` : str name of Instance
        - `node labels` : N-array of the node labels from 0 to N-1
        - `parameter names`:N-array of parameter names 
        - `parameters`:  array of size P corresponding paramter values. Same size as `parameter names`
        - `coordinate system`: array of coordinate system used
        - `dimension` : dimension of the system
        - `time` : array of size T of each time point found in the step
        - `output vars`: array of size O. Array of names of output variables
        - `data` : [T,N,O] shaped array of the data from the simulation
        - `headers` 
        - `input vars`
        - `export`
        - `nodes` 
        - `elements`




    '''
    
    # If no save_path is given use odb name
    if save_path is None:
        odb_path = Path(odb_name)
        save_path = odb_path.parent/f'{odb_path.stem}.mat'

    # Create temporary Json files to parse to Abaqus CAE
    with open('field_outputs.json','w') as f:
        assert isinstance(field_outputs,dict)
        json.dump(field_outputs,f) 
    with open('parameters.json','w') as f:
        
        assert isinstance(parameters,dict)
        json.dump(parameters,f) 


    #Check that we are passing abaqus command
    if 'abq' not in abaqus_version and 'abaqus' not in abaqus_version:
        raise ValueError('Only abaqus commands can be used either abaqus or abq<version> (e.g. abq2024)')

    result = subprocess.run([f'{abaqus_version}','cae',f'noGui=get_results_Abaqus_API.py', '--',
                            'vars_start' ,odb_name,instanceName,'field_outputs.json',stepName,'parameters.json',save_path],
                            shell = True,capture_output=True,text=True)
    

    if debug:
        print(f'The Return Code returned with exit code {result.returncode}')
        return result
    else:
        rpys = glob.glob('abaqus.rpy*')
        for rpy in rpys:
            os.remove(rpy)

    if result.returncode == 0:

        if delete_json:
            os.remove('field_outputs.json')
            os.remove('parameters.json')

        if return_mat:
            return loaddata(save_path)
    else:
        
        raise ValueError(f'Subprocess returned an Exit code of {result.returncode}, Run debug = True to return result')

if __name__ =='__main__':
    x = get_ODB_results(
    instanceName = 'TUBE-1',
    field_outputs = {'U':('U1','U2','U3')},
    stepName = 'TubeCrush',
    parameters = {'p':50.,'q':10.},
    save_path = 'test.json',
    odb_name = 'TubeCrush.odb',
    abaqus_version = 'abq2024hf3', 
    )