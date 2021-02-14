#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:15:14 2020

@author: mike_ubuntu
"""

import time
import matplotlib.pyplot as plt
import numpy as np

class Training_History(object):
    def __init__(self, filename = None):
        self._fitness_history = []
        self.time_init = time.time()
        self.filename = filename

    def reset(self):
        self._fitness_history = []
        self.time_init = time.time()
    
    def reset_timer(self):
        self.time_init = time.time()
    
    @property
    def runtime(self):
        return time.time() - self.time_init
    
    @property
    def iteration_history(self):
        return self._iteration_history
    
    @iteration_history.setter
    def iteration_history(self, hist):
        self._iteration_history = hist    
    
    @property
    def fitness_history(self):
        return self._fitness_history
    
    @property
    def achieved_fitness(self):
        return self._fitness_history[-1]
    
    @property
    def equation(self):
        return self._equation

    @equation.setter
    def equation(self, eq):
        self._equation = eq
    
    def extend_fitness_history(self, new):
        if type(new) == list or type(new) == tuple:
            self._fitness_history.extend(new)
        else:
            self._fitness_history.append(new)
        
    def save_fitness(self):
        if type(self.filename) == type(None):
            pass
            #raise ValueError('File name for the fitness function export not defined')
        else:
            np.save(self.filename, self._fitness_history)


class Visual(object):
    def __init__(self, n_iter):
        plt.ion()
        self.x_max = n_iter
        
    def launch(self):
        self.fig, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [])
        self.ax.grid()
        self.x_counter = 0

        self.x_history = []
        self.y_history = []
        
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(-1, self.x_max)        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()    
        
    def update(self, y_data):
        self.x_history.append(self.x_counter)
        self.y_history.append(y_data)
        self.x_counter += 1
        
        self.lines.set_xdata(self.x_history)
        self.lines.set_ydata(self.y_history)

        self.ax.relim()
        self.ax.autoscale_view()   
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()        