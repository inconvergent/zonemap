# -*- coding: utf-8 -*-

cimport cython

cdef inline void long_array_init(long *a,long n,long v) nogil:
  """
  initialize longeger array a of length n with longeger value v
  """
  cdef long i
  for i in xrange(n):
    a[i] = v
  return

cdef inline void double_array_init(double *a, long n, double v) nogil:
  """
  initialize double array a of length n with double value v
  """
  cdef long i
  for i in xrange(n):
    a[i] = v
  return

