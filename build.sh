#!/bin/sh
PYTHONPATH=.:./pypy pypy/rpython/bin/rpython -O2 targetmicrograd.py
