# -*- coding: utf-8 -*-
"""Experimental variants for the SSC module. They contain chosen prompts
and categories assigned to them.
"""

def mode1():
    """phonems vs phonems"""
    mode = {
        '/diy/' : '/diy/',
        '/iy/' : '/iy/',
        '/m/' : '/m/',
        '/n/' : '/n/',
        '/piy/' : '/piy/',
        '/tiy/' : '/tiy/',
        '/uw/' : '/uw/'}
    return mode

def mode2():
    """words vs words"""
    mode = {
        'gnaw' : 'gnaw',
        'knew' : 'knew',
        'pat' : 'pat',
        'pot' : 'pot'}
    return mode

def mode3():
    """words vs phonems"""
    mode = {
        '/diy/' : 'phoneme',
        '/iy/' : 'phoneme',
        '/m/' : 'phoneme',
        '/n/' : 'phoneme',
        '/piy/' : 'phoneme',
        '/tiy/' : 'phoneme',
        '/uw/' : 'phoneme',
        'gnaw' : 'word',
        'knew' : 'word',
        'pat' : 'word',
        'pot' : 'word'}
    return mode

def mode4():
    """free for all"""
    mode = {
        '/diy/' : '/diy/',
        '/iy/' : '/iy/',
        '/m/' : '/m/',
        '/n/' : '/n/',
        '/piy/' : '/piy/',
        '/tiy/' : '/tiy/',
        '/uw/' : '/uw/',
        'gnaw' : 'gnaw',
        'knew' : 'knew',
        'pat' : 'pat',
        'pot' : 'pot'}
    return mode

def mode5():
    """phonological division, gnaw/knew vs pat/pot"""
    mode = {
        'gnaw' : 'gnaw/knew',
        'knew' : 'gnaw/knew',
        'pat' : 'pat/pot',
        'pot' : 'pat/pot'}
    return mode

def mode6():
    """phonological division, gnaw/pot vs pat/knew""" 
    mode = {
        'gnaw' : 'gnaw/pot',
        'knew' : 'knew/pat',
        'pat' : 'knew/pat',
        'pot' : 'gnaw/pot'}
    return mode

def mode7():
    """fphonological division, /diy//iy/ vs /m//n/"""
    mode = {
        '/diy/' : '/diy//iy/',
        '/iy/' : '/diy//iy/',
        '/m/' : '/m//n/',
        '/n/' : '/m//n/',}
    return mode

def mode8():
    """fphonological division, /m/ vs /tiy/ vs /uw/ vs pot"""
    mode = {
        '/m/' : '/m/',
        '/tiy/' : '/tiy/',
        '/uw/' : '/uw/',
        'pot' : 'pot'}
    return mode


    