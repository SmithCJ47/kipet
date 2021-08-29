#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is where all of the multiprocessing stuff will be held until it works

"""

from multiprocessing import Queue, Process

class Multiprocess(object):

    def __init__(self, func):
        self.func = func
        self.processes = {}
        self.queuese = {}
        self.results = {}

    def __call__(self, *args, **kwargs):
        
        num_processes = kwargs.get('num_processes', 2)
        plist = list(range(1, num_processes+1))
        for i in plist:
            q = Queue()
            self.queuese[i] = q
            p = Process(target=self.func, args = tuple([q, i, *args]))
            self.processes[i] = p
            p.start()

        for i in plist: # check the ordr for join()
            self.results[i] = self.queuese[i].get()

        for i in plist:
            self.processes[i].join()

        return self.results
        