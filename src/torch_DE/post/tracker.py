from pandas import DataFrame,merge
from matplotlib import pyplot as plt
import torch
class Tracker(dict):
    def log(self,epoch,*,to_log:dict = None,**kwargs):
        if to_log is None:
            to_log = kwargs
        elif isinstance(to_log,dict):
            to_log.update(kwargs)
        else: raise ValueError(f'Expected variable to_log to be type dict got type {type(to_log)} instead')

        with torch.no_grad():
            for key,val in to_log.items():
                if key not in self.keys():
                    self[key] = []
                if isinstance(val,torch.Tensor):
                    val = val.cpu()
                self[key].append((epoch,float(val)))

    def to_DataFrame(self,fillna = None):
        '''
        Convert logger into dataframe
        '''
        merge_df = DataFrame(columns=['epoch'])
        for col,data in self.items():
            df = DataFrame(data,columns = ['epoch',col])
            merge_df = merge(merge_df,df,how ='outer',on = 'epoch')

        if fillna is not None:
            merge_df.fillna(fillna)
        return merge_df

    def to_csv(self,filename,*,fillna = None):
        df = self.to_DataFrame(fillna)
        df.to_csv(filename)

    def plot(self,key:str,*,show = True,save_name = None,output = False,y_scale = 'linear',x_scale = 'linear',**plot_kwargs):
        df = DataFrame(self[key],columns = ['epoch',key])
        x,y = df['epoch'].to_numpy(),df[key].to_numpy()

        if show is False and save_name is None:
            if output:
                return (x,y),('epoch',key)
            else:
                raise ValueError('Atleast one of the keywords show, save_name and output must be True or not None')
        
        fig,ax = plt.subplots()
        ax.plot(x,y,**plot_kwargs)
        ax.set_title(f'Plot of {key} over epochs')
        ax.set_xlabel('epochs')
        ax.set_ylabel(key)

        ax.set_yscale(y_scale)
        ax.set_xscale(x_scale)

        if show:
            fig.show()
        
        if save_name is not None:
            fig.savefig(save_name)
        
        plt.close(fig)
        if output:
            return (x,y),('epoch',key)



