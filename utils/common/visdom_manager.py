import torch
import visdom
import numpy as np

class VisObj:
    def __init__(self, viz, env, win, legend, opts):
        self.viz = viz
        self.env = env
        self.win = win
        self.legend = legend
        self.opts = opts
        self.inserted = False
        
    def validate_input_(self, X):
        if isinstance(X, np.ndarray):
            return X
        elif isinstance(X, int) or isinstance(X, float) or isinstance(X, np.float32):
            return np.array([X])
        elif isinstance(X, torch.Tensor):
            X = X.cpu().data.numpy()
            if X.ndim == 0:
                X = X.reshape((1))
            return X
        
    def update(self, X, Y):
        if self.viz.get_window_data(win=self.win, env=self.env) == '':
            update_type = None  
        elif self.inserted:
            update_type = 'append'  
        else:
            update_type = 'new'
        self.viz.line(X=self.validate_input_(X), Y=self.validate_input_(Y), 
                      env=self.env, win=self.win, name=self.legend,
                      opts=self.opts, update=update_type)
        self.inserted = True

    def clear(self):
        self.viz.line(X=None, Y=None, env=self.env, win=self.win, name=self.legend, update='remove')
    
    def close(self):
        self.viz.close(env=self.env, win=self.win)
    
    def __repr__(self):
        return '>> Visdom Object with env:{} win:{} legend:{}\n'.format(self.env, self.win, self.legend)
 
class VisManager:    
    def __init__(self, env, win_pref='', targets={}, host='localhost', port='8097', enable_log=False):
        server = 'http://{}'.format(host)
        self.viz = visdom.Visdom(server=server, port=port)
        self.env = env
        assert self.viz.check_connection(), 'Visdom server is not active on server {}:{}'.format(server, port)  
        print('Visdom server connected on {}:{}'.format(server, port))
              
        # Appending to previous log
        if enable_log:
            self.log_win = '{}_log'.format(win_pref)
            prev_txt = self.viz.get_window_data(env=self.env, win=self.log_win)
            if prev_txt == '':
                self.txt = prev_txt
            else:
                import json
                self.txt = json.loads(prev_txt)['content']

        # Initialize all target windows
        self.win_pool = {}
        for win in targets:
            self.win_pool[win] = {}
            opts = self.get_default_opts_(win)
            for legend in targets[win]:
                visobj = VisObj(self.viz, self.env, win, legend, opts)
                self.win_pool[win][legend] = visobj
                print('Initialize {}'.format(str(visobj)))
    
    def get_default_opts_(self, title):
        layout = {'plotly': dict(title=title, xaxis={'title': 'epochs'})}
        opts=dict(mode='lines', showlegend=True, layoutopts=layout)
        #opts=dict(mode='marker+lines', 
        #      markersize=5,
        #      markersymbol='dot',
        #      markers={'line': {'width': 0.5}},
        #      showlegend=True, layoutopts=layout)
        return opts

    def get_win(self, win, legend):
        return self.win_pool[win][legend] 

    def save_state(self):
        self.viz.save(envs=[self.env])
        
    def clear_all(self):
        for win in self.win_pool:
            for legend in self.win_pool[win]:
                self.win_pool[win][legend].clear()

    def log(self, txt):
        self.txt += '{}<br>'.format(txt)
        self.viz.text(text=self.txt, env=self.env, win=self.log_win)

    def print_(self):
        print('Visdom Manager Window Pool:\n')
        for win in self.win_pool:
            for legend in self.win_pool[win]:
                print(self.win_pool[win][legend])

class DummyVisObj:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass

class DummyVisManager:
    def __init__(self, *args, **kwargs):
        pass
    
    def save_state(self):
        pass
