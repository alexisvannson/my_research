#!/usr/bin/env python
# -*- coding: utf-8 -*-

__copyright__ = '(c) Magoules Research Group, 1996. All Rights Reserved.'
__date__ = '2024-06-01'

# Python packages
import sys

# MRG packages
verbose0 = True
verbose1 = True
verbose2 = True
verbose3 = True


def get_execution_verbosity():
    """
    Get execution levels.

    Parameters
    ----------

    Returns
    -------
    verbose0 : boolean
        -- to fix
    verbose1 : boolean
        -- for text (if True activate, if False disable)
    verbose2 : boolean
        -- to fix
    verbose3 : boolean
        -- to fix

    """

    return verbose0, verbose1, verbose2, verbose3


def set_execution_verbosity(level0=False, level1=False, level2=False, level3=False):
    """
    Set execution levels.

    Parameters
    ----------
    level0 : boolean
        -- True or False
    level1 : boolean
        -- True or False
    level2 : boolean
        -- True or False
    level3 : boolean
        -- True or False

    Returns
    -------
    None : NoneType
        -- None

    """
    global verbose0
    global verbose1
    global verbose2
    global verbose3

    verbose0 = level0
    verbose1 = level1
    verbose2 = level2
    verbose3 = level3

    return


def handle_sys_getframe():
    """
    Handle system getframe.

    Parameters
    ----------

    Returns
    -------
    None : NoneType
        -- None

    """
    this_function_name = sys._getframe(1).f_code.co_name
    this_line_number = sys._getframe(1).f_lineno
    this_filename = sys._getframe(1).f_code.co_filename
    err_msg = f'\nTraceback (most recent call last):'
    err_msg += f'\n  File {this_filename}, line {this_line_number}, in {this_function_name}'

    return err_msg
