#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is where all of the multiprocessing stuff will be held until it works

"""

from multiprocessing import Queue, Process#, Lock

# lock = Lock()

class Multiprocess(object):

    def __init__(self, func):
        
        self.func = func
        self.processes = {}
        self.queues = {}
        self.results = {}

    def __call__(self, *args, **kwargs):
        
        num_processes = kwargs.get('num_processes', 2)
        plist = list(range(1, num_processes+1))
        q = Queue()
        
        for i in plist:    
            self.queues[i] = q
            p = Process(target=self.func, args=tuple([q, i, *args]))
            self.processes[i] = p
            p.start()

        for i in plist: # check the ordr for join()
            self.results[i] = self.queues[i].get()

        for i in plist:
            self.processes[i].join()
            
        return self.results
    
    # def lock_func(self, *args, **kwargs):
        
    #     lock.acquire()
    #     try:
    #         self.func(*args, **kwargs)
    #     finally:
    #         lock.release()  
        
      
      
      
      
      
      
      