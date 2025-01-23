import numpy as np
from scipy.io import loadmat
import os
import shutil
import glob

def moveFiles(new_dir,filename):
    # Ensure the destination folder exists
    os.makedirs(new_dir, exist_ok=True)
    files = glob.glob(filename)

    for file in files:
        shutil.move(file,new_dir)


#We need to change
def loaddata(filename,print_keys = False,**kwargs):
    '''
    This function helps load in .mat files by cleaning up string arrays as all strings must be the same length for each string element so trailing whitespaces 
    are added to pad smaller strings. Changes the string arrays to list of strings

    Note that np vectors of size (n,) are converted to size (n,1) from loadmat

    Other than that work identical to loadmat from scipy
    '''
    data = loadmat(filename,**kwargs)

    #String Arrays 
    for key in ['headers','parameter names','output vars','input vars']:
        data[key] = [s.strip() for s in data[key]]
        if print_keys:
            print(data[key])
    #Strings 
    for key in ['instance name','step','export']:
        data[key] = str(data[key][0])

    #Arrays
    for key in ['node labels','time','parameters']:
        assert key in data.keys()
        data[key] = data[key].squeeze()
    return data

# data = loaddata('tubecrush.mat')


def convert_compact_to_tabular(results:dict):
    '''
    Convert a dictionary in compact form to tabular form. This mainly changes the 'data' item. This form is much more useful for parametric deep learning
    '''
    if results['export'] == 'tabular':
        return results

    node_labels = results['node labels']
    coords= results['coords']
    cart_csys = results['coordinate system']
    time_array= results['time']
    data = results['data']
    output_vars = results['output vars']
    if results['parameters']:
        params_names,parameters = results['parameter names'],results['parameters']
        parameters = [np.ones((coords.shape[0],1))*param for param in parameters]
    else:
        params_names,parameters = [], []
    frame_csv =  np.concatenate ( [ np.concatenate([node_labels[:,np.newaxis],*parameters,coords,np.ones((coords.shape[0],1))*t,data[i]],axis = -1) for i,t in enumerate(time_array) ], axis = 0,dtype= np.float32)
    
    input_vars = ['node label',*params_names,*cart_csys,'t'] 
    headers = input_vars + output_vars    
    
    
    tabular_results = {}
    # Copy all data except for data

    for key,value in results.items():
        if key != 'data':
            tabular_results[key] = value

    tabular_results['data'] = frame_csv
    tabular_results['headers'] = headers
    tabular_results['input vars'] = input_vars
    tabular_results['export'] = 'tabular'
    return tabular_results


def convert_tabular_to_compact(results:dict):
    '''
    Convert a dictionary in tabular form to compact form. 
    This mainly changes the 'data' item. This form is much more useful non-parametric/graph based deep learning
    Also this is useful for converting back to an format more easily read by abaqus
    '''

    if results['export'] == 'compact':
        return results


    n_nodes = results['node labels'].shape[-1]
    n_points = results['data'].shape[0]
    assert n_points % n_nodes == 0

    headers = results['headers']
    output = results['data'][:,[headers.index(output_var) for output_var in results['output vars']]  ]
    
    time_points = results['time'].shape[-1]

    #Reshape so its time,node_label,output_var
    output = output.reshape(time_points,n_nodes,output.shape[-1])
    results['data'] = output
    results['export'] = 'compact'
    results['headers'] = [s for s in results['output vars']]
    results['input vars'] = []
    # Output Vars still remain the same
    return results

