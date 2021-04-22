#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:53:24 2020

@author: mike_ubuntu
"""
import time
import numpy as np
import tensorflow as tf
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1))):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots()
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()


def approximator_shallow(*args):
#    hidden_layers = 100
#    input_layers = data.shape[-1]
#    output_dim = 1
    data = tf.stack(args, axis = 1, name = 'data_stacked')
#    print(data)
    with tf.name_scope("UA_fit"):   
        w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
        b1 = tf.get_default_graph().get_tensor_by_name("b1:0")
        w_out = tf.get_default_graph().get_tensor_by_name("w_out:0")        
        
        ws1 = tf.matmul(data, w1) + b1
        a = tf.nn.sigmoid(ws1)
        ws_out = tf.matmul(a, w_out)
    return ws_out

def approximator_deep(*args):

#    hidden_layers = 100
#    input_layers = data.shape[-1]
#    output_dim = 1
    data = tf.stack(args, axis = 1, name = 'data_stacked')
    data = tf.squeeze(data)
#    print(data)
    with tf.name_scope("UA_fit"):   
        w1 = tf.get_default_graph().get_tensor_by_name("w1:0")
        b1 = tf.get_default_graph().get_tensor_by_name("b1:0")
        w2 = tf.get_default_graph().get_tensor_by_name("w2:0")
        b2 = tf.get_default_graph().get_tensor_by_name("b2:0")
        w3 = tf.get_default_graph().get_tensor_by_name("w3:0")
        b3 = tf.get_default_graph().get_tensor_by_name("b3:0")
        w4 = tf.get_default_graph().get_tensor_by_name("w4:0")
        b4 = tf.get_default_graph().get_tensor_by_name("b4:0")
        w5 = tf.get_default_graph().get_tensor_by_name("w5:0")
        b5 = tf.get_default_graph().get_tensor_by_name("b5:0")        
        w_out = tf.get_default_graph().get_tensor_by_name("w_out:0")
        
        ws1 = tf.matmul(data, w1) + b1
        a1 = tf.nn.sigmoid(ws1) #tf.nn.sigmoid(ws1)
        
        ws2 = tf.matmul(a1, w2) + b2
        a2 = tf.nn.sigmoid(ws2) #tf.nn.sigmoid(ws2)
        
        ws3 = tf.matmul(a2, w3) + b3
        a3 = tf.nn.sigmoid(ws3) #tf.nn.sigmoid(ws3)
        
        
        ws4 = tf.matmul(a3, w4) + b4
        a4 = tf.nn.sigmoid(ws4) #tf.nn.sigmoid(ws3)
        
        ws5 = tf.matmul(a4, w5) + b5
        a5 = tf.nn.sigmoid(ws5) #tf.nn.sigmoid(ws3)
        
        ws_out = tf.matmul(a5, w_out)
    return ws_out                   
        
def approximate_ann_shallow(data):
    hidden_layers = 50
    input_layers = data.shape[-1]
    output_dim = 1
    
    with tf.name_scope("UA_fit"):
        w1 = tf.get_variable(name = "w1", dtype = tf.float32, shape=[input_layers, hidden_layers], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b1 = tf.get_variable(name = "b1", dtype = tf.float32, shape=[hidden_layers], 
                            initializer=tf.constant_initializer(0.))
        
        ws1 = tf.matmul(data, w1) + b1
        a = tf.nn.sigmoid(ws1)
        
        w_out = tf.get_variable(name = "w_out", dtype = tf.float32, shape=[hidden_layers, output_dim], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        ws_out = tf.matmul(a, w_out)
#    print(w1.name)        
    return ws_out    

        
def approximate_ann_deep(data):
    hidden_layers_1 = 256
    hidden_layers_2 = 512
    hidden_layers_3 = 256
    hidden_layers_4 = 128
    hidden_layers_5 = 64
    
    input_layers = data.shape[-1]
    output_dim = 1
    
    with tf.name_scope("UA_fit"):
        w1 = tf.get_variable(name = "w1", dtype = tf.float32, shape=[input_layers, hidden_layers_1], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b1 = tf.get_variable(name = "b1", dtype = tf.float32, shape=[hidden_layers_1], 
                            initializer=tf.constant_initializer(0.))
        
        w2 = tf.get_variable(name = "w2", dtype = tf.float32, shape=[hidden_layers_1, hidden_layers_2], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b2 = tf.get_variable(name = "b2", dtype = tf.float32, shape=[hidden_layers_2], 
                            initializer=tf.constant_initializer(0.))
        
        w3 = tf.get_variable(name = "w3", dtype = tf.float32, shape=[hidden_layers_2, hidden_layers_3], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b3 = tf.get_variable(name = "b3", dtype = tf.float32, shape=[hidden_layers_3], 
                            initializer=tf.constant_initializer(0.))
        
        w4 = tf.get_variable(name = "w4", dtype = tf.float32, shape=[hidden_layers_3, hidden_layers_4], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b4 = tf.get_variable(name = "b4", dtype = tf.float32, shape=[hidden_layers_4], 
                            initializer=tf.constant_initializer(0.))
        
        w5 = tf.get_variable(name = "w5", dtype = tf.float32, shape=[hidden_layers_4, hidden_layers_5], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        b5 = tf.get_variable(name = "b5", dtype = tf.float32, shape=[hidden_layers_5], 
                            initializer=tf.constant_initializer(0.))
        
        
        ws1 = tf.matmul(data, w1) + b1
        a1 = tf.nn.sigmoid(ws1)
        
        ws2 = tf.matmul(a1, w2) + b2
        a2 = tf.nn.sigmoid(ws2)
        
        ws3 = tf.matmul(a2, w3) + b3
        a3 = tf.nn.sigmoid(ws3)
        
        ws4 = tf.matmul(a3, w4) + b4
        a4 = tf.nn.sigmoid(ws4)
        
        ws5 = tf.matmul(a4, w5) + b5
        a5 = tf.nn.sigmoid(ws5)        
        
        w_out = tf.get_variable(name = "w_out", dtype = tf.float32, shape=[hidden_layers_5, output_dim], 
                            initializer=tf.random_normal_initializer(stddev = 1.))
        ws_out = tf.matmul(a5, w_out)

#    print(w1.name)        
    return ws_out    

class Differentiable_Function(object):
    def __init__(self, function):
        self.function = function
        
    def differentiate(self, data, axis_names, orders = 1):
        print(type(data))
        gradient = tf.gradients(ys=self.function, xs=data)
        if orders == 1:
            deriv_history = [[der_axis,].extend(self.deriv_history) for der_axis in axis_names]
            return (gradient, deriv_history)
        else:
            derivatives = []; deriv_history = []
            for idx in len(data):
                temp_fun = Differentiable_Function(gradient[idx])
                der_fun, history = temp_fun.differentiate(data, axis_names, orders = orders - 1)
                derivatives.extend(der_fun); deriv_history.extend(history)
            return derivatives, deriv_history

class Approximator(object):
    def __init__(self, data, steps, *kwargs):
        self.training_session = tf.Session()
        self.data = data; 
        self.steps = steps
        
        self.data_grid = np.indices(self.data.shape)
        self.dgr = np.transpose(self.data_grid.reshape((self.data_grid.shape[0], np.prod(self.data_grid.shape[1:]))))
        
        self.coords = [self.data_grid[idx]*self.steps[idx] for idx in np.arange(len(self.steps))]
        self.coords = np.stack(self.coords, axis=-1)
        
        self.set_interp_function(self.coords, self.data)

    def grid_function(self, coords, coord_eps = 1e-10):
        for idx, vals in np.ndenumerate(self.coords[..., 0]): 
            if np.all(np.abs(self.coords[idx] - np.array(coords)) < coord_eps):
                return self.data[idx]        
        return -np.inf        
        
    def train(self, batch_proportion = 0.9, epochs = 100):
#        coords = np.meshgrid(*[np.arange(data.shape[i]) for i in np.arange(np.ndim(data))], indexing = 'ij')
#        coords = np.stack(coords, axis=-1)
        
        
        x = tf.placeholder(tf.float32, shape = [None, self.data.ndim], name="x")
        y_appr = approximate_ann_deep(x)
        y_true = tf.placeholder(tf.float32, shape = [None, 1], name="y_true")  #self.interp_function(x)
#        y_true_var = tf.Variable(y_true)
#        y_true = tf.sin(x)
        
        with tf.variable_scope('Loss'):
            loss = tf.losses.mean_squared_error(y_true, y_appr)
#            loss_summary_sc = tf.summary.scalar('loss', loss) 

        adam = tf.train.AdamOptimizer(learning_rate=1e-2)
        train_optimizer = adam.minimize(loss)

        self.training_session.run(tf.global_variables_initializer())        
        for epoch in np.arange(epochs):
            random_indices = self.dgr[np.random.choice(np.arange(self.dgr.shape[0]), size= np.int(self.dgr.shape[0]*batch_proportion), replace= False)]
            x_eval = np.matrix([self.coords[tuple(idx)] for idx in random_indices])
#            print(x_eval.shape)
#            print(x_eval)
#            x_eval = x_eval.flatten()
            y_eval =  [self.interp_function(self.coords[tuple(idx)]) for idx in random_indices] #k * x_eval[:, 0] #np.array()
            y_eval = np.asarray(y_eval).reshape((len(y_eval), 1))#  np.squeeze()
            print(y_eval.shape)
#            if epoch == epochs - 1:
#                x_out = np.squeeze(np.asarray(x_eval))
#                plt.scatter(x_out, y_eval)
#            print(y_eval)
            
#            y_eval_tensor = tf.convert_to_tensor(y_eval, dtype = np.float32)
#            y_eval = y_eval.reshape((y_eval.size))
#            print(y_eval.shape)
            
#            print(list(zip(x_eval[:10], y_eval[:10])))
#            print(y_eval)
            feed = {x: x_eval, y_true: y_eval}
            current_loss, _, x_cur, y_cur, y_approx = self.training_session.run([loss, train_optimizer,x, y_true, y_appr], feed_dict = feed)
            print(epoch, ' loss: ', current_loss)
#            print(list(zip(x_cur[:10], y_cur[:10])))
#            print('y', y_cur)
#            print('y_approx', y_approx)
#            print('weights 1', w1)
#    
#    @property        
#    def derivatives(self, order):
#        self.inputs = self.coords.reshape((np.prod(self.coords.shape[:-1]), self.coords.shape[-1]))
#        self.y_res = self.training_session.run([y_appr], feed_dict={
#                x: self.inputs
#            })
#        self.y_res = self.y_res.reshape(self.data.shape)
        
#    def differentitate(self, function, coordinate):
#        
    
    def interp_function(self, coords):
        try:
            return self._interp(coords)[0]
        except IndexError:
            return np.float(self._interp(coords))
        
    def set_interp_function(self, data_coords, data, eps = 1e-9):
        data_coords = [np.linspace(0, self.steps[idx]*(self.data.shape[idx] - 1), self.data.shape[idx]) for idx in np.arange(self.data.ndim)]
        for dim in np.arange(len(data_coords)):
            data_coords[dim][0] -= eps; data_coords[dim][-1] += eps
                
        print(data_coords)
        self._interp = RegularGridInterpolator(data_coords, data, method = 'nearest')
        print('interpolator set')
    
    
if __name__ == '__main__':
    
    def ic_1(x):
        x_max = 5.
        return np.sin(x/x_max*np.pi)*np.sin((5-x)/x_max*np.pi)
    
    
    def ic_2(x):
        x_max = 5.; coeff = -1
        return coeff * np.sin(x/x_max*np.pi)*np.sin((5-x)/x_max*np.pi)
    
    
    
    x_shape = 301; t_shape = 301
    x_max = 5; t_max = 1
    x_vals = np.linspace(0, x_max, x_shape)
    t_vals = np.linspace(0, t_max, t_shape)
    delta_x = x_vals[1] - x_vals[0]; delta_t = t_vals[1] - t_vals[0]
    k = 0.5
    
    solution = np.empty((t_shape, x_shape))
    solution[:, 0] = solution[:, -1] = 0
    solution[0, :] = ic_1(x_vals)
    solution[1, :] = solution[0, :] + delta_t * ic_2(x_vals)
    
    for t in np.arange(2, t_shape):
        for x in np.arange(1, x_shape - 1):
            solution[t, x] = k*delta_t**2/delta_x**2 * (solution[t-1, x+1] - 2*solution[t-1, x] + solution[t-1, x-1]) + 2*solution[t-1, x] - solution[t-2, x]
    
    
    tf.reset_default_graph()
    x = np.linspace(0, 5, 301)
    y = np.linspace(0, 5, 301)
    xx, yy = np.meshgrid(x, y)
    
    z = np.sin(xx)*np.sin(yy)*np.sin(4-xx)*np.sin(4-yy) #+ np.cos(yy + np.pi/4)#
    test_approx = Approximator(z, (x[1] - x[0], y[1] - y[0]))
    test_approx.train(batch_proportion = 0.8, epochs=600)
    z_approximation = np.array(test_approx.y_res)
    z_approximation  = z_approximation.reshape(z_approximation.size)
    inputs = test_approx.coords.reshape((np.prod(test_approx.coords.shape[:-1]), test_approx.coords.shape[-1]))
    
    dims = ['x', 'y', 'z', 't']
    coords = []
    for dim_idx in range(test_approx.dgr.shape[-1]):
        var_name = dims[dim_idx] + '_coords'
        coords.append(tf.placeholder(dtype = tf.float32, shape = (test_approx.dgr.shape[0],), name = var_name))
        
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(coords)
        approx = approximator_deep(*coords)
    dz_dx = tape.gradient(approx, sources = coords)
    
    gradient = []
    for dim_idx in range(test_approx.dgr.shape[-1]):
        gradient.append(dz_dx.eval(session = test_approx.training_session,feed_dict={x_coords:inputs[:, dim_idx]}))
    
#    Heatmap(temp)
        