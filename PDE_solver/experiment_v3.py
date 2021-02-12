# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 08:26:39 2020

@author: Sashka
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from scipy.optimize import minimize
# from scipy.optimize import differential_evolution
# from scipy.optimize import dual_annealing
import numba
import time
import matplotlib.pyplot as plt

from PDE_solver import solution_interp
from PDE_solver import solution_interp_RBF
from PDE_solver import solution_interp_nn
from PDE_solver import string_reshape
from PDE_solver import operator_norm
import ray




plt.rcParams["figure.max_open_warning"] = 1000


norm_bond = float(100)

@ray.remote
def init_field_expetiment(nrun,grid_scenario=[[10,10]],interp_type='random',diff_scheme=[1,2]):
    np.random.seed()
    experiment=[]
    
    x = np.linspace(0, 1, 10)
    t = np.linspace(0, 1, 10)
    
    old_grid = numba.typed.List()
    
    old_grid.append(x)
    old_grid.append(t)
    
    arr = np.random.random((10, 10))
    
    for grid_res in grid_scenario:
    
        
        x = np.linspace(0, 1, grid_res[0])
        t = np.linspace(0, 1, grid_res[1])
    
        new_grid = numba.typed.List()
    
        new_grid.append(x)
        new_grid.append(t)
    
        bcond = [{'boundary': 0, 'axis': 0, 'string': np.zeros(len(new_grid[0]))},
              {'boundary': -1, 'axis': 0, 'string': np.zeros(len(new_grid[0]))},
              {'boundary': 0, 'axis': 1, 'string': np.sin(np.pi * new_grid[1])},
              {'boundary': -1, 'axis': 1, 'string': np.sin(np.pi * new_grid[1])}]
    

        if interp_type=='scikit_interpn':
            arr=solution_interp(old_grid,arr,new_grid)
        elif interp_type=='interp_RBF':
            arr=solution_interp_RBF(old_grid,arr,new_grid,method='linear',smooth=10)
        elif interp_type=='interp_nn':
            arr=solution_interp_nn(old_grid,arr,new_grid)
        else:
            arr = np.random.random((len(x), len(t)))
            
        exact_sln = f'exact_sln/wave_{grid_res[0]}.csv'
    
        wolfram_interp =  np.genfromtxt(exact_sln, delimiter=',')
        
        start_time = time.time()
        
        operator_norm_1= lambda x: operator_norm(x, new_grid,[[(1, 0, 2, 1)], [(-1 / 4, 1, 2, 1)]], norm_bond, bcond,scheme_order=diff_scheme[0],boundary_order=diff_scheme[1]) 
        
        opt = minimize(operator_norm_1, arr.reshape(-1),options={'disp': False, 'maxiter': 3000}, tol=0.05)
        
        elapsed_time = time.time() - start_time
        
        print(f'[{datetime.datetime.now()}] grid_x = {grid_res[0]} grid_t={grid_res[1]} time = {elapsed_time}')
    
        full_sln_interp= string_reshape(opt.x, new_grid)
        
        
        error = np.abs(full_sln_interp - wolfram_interp)
        max_error = np.max(error)
        wolfram_MAE = np.mean(error)
        
        arr=full_sln_interp
        
        # plot_3D_surface(full_sln_interp, wolfram_interp, new_grid)
        
        experiment.append({'grid_x':len(x),'grid_t':len(t),'time':elapsed_time,'MAE':wolfram_MAE,'max_err':max_error,'interp_type':interp_type,'scheme_order':diff_scheme[0],'boundary_order':diff_scheme[1]})
        
        old_grid=new_grid
        
    return experiment

if os.path.isfile("interp_v3.csv"): 
    df=pd.read_csv("interp_v3.csv",index_col=0)
else:
    df=pd.DataFrame(columns=['grid_x','grid_t','time','MAE','max_err','interp_type','scheme_order','boundary_order'])



ray.init(ignore_reinit_error=True)

#warm up
results = ray.get([init_field_expetiment.remote(run,interp_type='interp_nn') for run in range(1)])



for ds in [[1,1],[2,2],[1,2]]:
    for interp_type in ['random','scikit_interpn','interp_RBF','interp_nn']:
        results = ray.get([init_field_expetiment.remote(run,interp_type=interp_type,grid_scenario=[[10,10],[15,15],[20,20],[25,25],[30,30],[35,35],[40,40],[45,45],[50,50]],diff_scheme=ds) for run in range(20)])
        for result in results:
            df=df.append(result,ignore_index=True)
        df.to_csv(f'interp_v3_{datetime.datetime.now().timestamp()}.csv',index=False)



    


