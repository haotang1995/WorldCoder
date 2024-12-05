#!/usr/bin/env python
# coding=utf-8

import os
import sys
import shutil
import time

# Save everything on the screen to a file
class Logger:
    def __init__(self, filename):
        if os.path.exists(filename):
            backup_filename = filename + '.' + time.strftime('%Y%m%d%H%M%S')
            shutil.move(filename, backup_filename)
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        message = message2level(message)
        #write to screen
        self.terminal.write(message)
        #write to file
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        os.fsync(self.log.fileno())

    def close(self):
        self.log.close()

def set_logger(filename):
    sys.stdout = Logger(filename)
    # sys.stderr = sys.stdout

GLOBAL_LEVEL = 0
def set_logging_level(level):
    global GLOBAL_LEVEL
    GLOBAL_LEVEL = level
def get_logging_level():
    return GLOBAL_LEVEL
def inc_logging_level():
    global GLOBAL_LEVEL
    GLOBAL_LEVEL += 1
def dec_logging_level():
    global GLOBAL_LEVEL 
    GLOBAL_LEVEL -= 1

PREV_NEWLINE = False
def message2level(message):
    global GLOBAL_LEVEL, PREV_NEWLINE
    if GLOBAL_LEVEL == 0: prefix = ''
    else: prefix = '  ' * (GLOBAL_LEVEL) + '| '

    if PREV_NEWLINE: prev_message = prefix 
    else: prev_message = ''
    out_message = prev_message + message.rstrip('\n').replace('\n', '\n' + prefix)
    new_lines = '\n' * (len(message) - len(message.rstrip('\n')))
    PREV_NEWLINE = new_lines != ''

    return out_message + new_lines

# Context manager to temporarily change the level
class LoggingLevel:
    def __init__(self, level=None,):
        self.old_level = get_logging_level()
        self.level = level if level is not None else self.old_level + 1
    def __enter__(self):
        set_logging_level(self.level)
    def __exit__(self, type, value, traceback):
        set_logging_level(self.old_level)
        return False
