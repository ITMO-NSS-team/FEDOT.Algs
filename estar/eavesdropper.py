#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 18:05:12 2020

@author: mike_ubuntu
"""

import copy
import pysftp
import numpy as np
import time
import matplotlib.pyplot as plt
import io

def readfile(sftp_connection, path):
    flo = io.BytesIO()
    sftp.getfo(path, flo)
    flo.seek(0)
    return np.load(flo)

class Visualizer(object):
    def __init__(self, epochs):
        plt.ion()
        self.x_max = epochs
        self.launched = False; 
        
    def set_grid_data(self, i, j, alpha):
        self.i = i; self.j = j; self.alpha = alpha
        
    def launch(self):
        self.launched = True
        
        self.fig, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [])
        self.ax.grid()
        self.x_counter = 0

        self.fig
        self.x_history = []
        self.y_history = []
        
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(-1, self.x_max)
        
        label = 'Fitness function for domain ' + str(self.i) + ' ' + str(self.j) + '; alpha index: ' + str(self.alpha)
        self.ax.set_title(label)        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()    
        
        
    def update(self, y_data):
        self.x_history = np.arange(y_data.size)
        self.y_history = y_data
        #self.x_counter += 1
        
        self.lines.set_xdata(self.x_history)
        self.lines.set_ydata(self.y_history)

        self.ax.relim()
        self.ax.autoscale_view()   
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
    def save_fig(self):
        figname = 'graphic_output/ff_'+str(self.i)+'_'+str(self.j)+'_'+str(self.alpha)+'.png'
        self.fig.savefig(figname)
        plt.close(self.fig)
        print('Saved file into ', figname)
        
        # Add procedure to save and close the figure
        
        
class Eavesdropper(object):
    def __init__(self, epochs, ping_period): #filename, 
        self.ping_period = ping_period
        self.epochs = epochs
        self.performed_reads = 0
        self.initiated = False
        
        self.preserve_figure_time = 30
        self.termination_timer = max([1, int(np.ceil(self.preserve_figure_time/ping_period))])
        self.terminated = False
        self.visualizer = Visualizer(self.epochs)
#        self.filename = filename
    
    def set_idx_data(self, idx, filename):
        self.visualizer.set_grid_data(idx[0], idx[1], idx[2])
        self.filename = filename
    
    def ping(self):
#        print('smth')
        if not self.terminated:
#            print(self.filename)
            try:
                current_data = readfile(sftp, self.filename)
            except FileNotFoundError:
                return
            self.performed_reads += 1
            print('read data from ', self.filename)
            print('timers', self.performed_reads, self.termination_timer)
            if not self.visualizer.launched:
                self.visualizer.launch()
                self.visualizer.update(current_data)
            else:
                self.visualizer.update(current_data)
                if self.performed_reads >= self.epochs:
                    self.termination_timer -= 1
                if self.termination_timer < 0:
                    self.visualizer.save_fig()
                    self.terminated = True
#        else:

        
    
if __name__ == '__main__':
    # Rough implementation: parameters are assumed beforehand
    epochs = 150
    alpha_range = 4; domains_x = 8; domains_y = 22
    
    #connection parameters
    
    cnopts = pysftp.CnOpts(); cnopts.hostkeys = None
    sftp = pysftp.Connection(host, username=username, password=password, cnopts = cnopts)
    path = '/home/akhvatov/ESTAR/graphic_output/'
    ping_period = 10 # seconds
    
    eavesdroppers = np.empty((domains_x, domains_y, alpha_range), dtype = object)
    for i in range(domains_x):
        for j in range(domains_y):
            for alpha in range(alpha_range):
                eavesdroppers[i, j , alpha] = Eavesdropper(epochs, ping_period)
    
    processed = [(2, 8)]
    
    for i in range(domains_x):
        for j in range(domains_y):
            for alpha in range(alpha_range):
                filename = path + 'ff_'+str(i)+'_'+str(j)+'_'+str(alpha)+'.npy'
                try:
                    sftp.remove(filename)
                except FileNotFoundError:
                    pass
#                print(filename)
                eavesdroppers[i, j, alpha].set_idx_data([i, j, alpha], filename)

    while True:
        print('ping!')
        terminated = np.empty(eavesdroppers.shape, dtype = object)
        for idx, ed in np.ndenumerate(eavesdroppers):
            if (idx[0], idx[1]) in processed or type(processed) == type(None):
                ed.ping()
                terminated[idx] = ed.terminated
            else:
                terminated[idx] = True
        if terminated.all():
            break
        time.sleep(ping_period)
    sftp.close()
    
    
#    for idx, ed in np.ndenumerate(eavesdroppers):
#        filename = path + 'ff_'+str(idx[0])+'_'+str(idx[1])+'_'+str(idx[2])+'.npy'
#        print(filename)
#        eavesdroppers[idx].set_idx_data(idx, filename)
#        print(ed.filename)
        
#        ed.launch()
                
#    ping_f = lambda x: x.ping()
#    ping = np.vectorize(ping_f)
#    
#    check_term_f = lambda x: x.terminated
#    check_term = np.vectorize(check_term_f)
        
    
#        for idx, ed in np.ndenumerate(eavesdroppers):
#            try:
#                ed.update(readfile(sftp, filenames[idx]))
#            except FileNotFoundError:
#                pass
        
        
                
            
