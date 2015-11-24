# -*- coding: utf-8 -*-
# cython: profile=True

from __future__ import division

cimport cython
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport sqrt

from helpers cimport long_array_init
from helpers cimport double_array_init

import numpy as np
cimport numpy as np

cdef long SIZE = 1024


cdef class Zonemap:

  def __init__(self, long nz):

    self.vnum = 0
    self.vsize = SIZE

    self.nz = nz
    self.total_zones = nz*nz
    self.greatest_zone_size = SIZE

    self.__init_zones()

    return

  def __cinit__(self, long nz, *arg, **args):

    cdef long total_zones = nz*nz

    self.VZ = <long *>malloc(SIZE*sizeof(long))

    self.Z = <sZ **>malloc(total_zones*sizeof(sZ*))

    return

  def __dealloc__(self):

    cdef long i

    for i in xrange(self.total_zones):

      free(self.Z[i].ZV)
      free(self.Z[i])

    free(self.Z)
    free(self.VZ)

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef void __init_zones(self) nogil:
    # somehow this did not work when executed inside cinit

    cdef long i
    cdef sZ *z

    for i in xrange(self.total_zones):

      z = <sZ *>malloc(sizeof(sZ))

      z.i = i
      z.size = SIZE
      z.count = 0
      z.ZV = <long *>malloc(SIZE*sizeof(long))

      self.Z[i] = z

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef void __assign_xy_arrays(self, double *X, double *Y) nogil:

    self.X = X
    self.Y = Y

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __add_vertex(self, long v1) nogil:
    """
    """

    cdef long vnum = self.vnum

    cdef double x = self.X[v1]
    cdef double y = self.Y[v1]

    cdef long z1 = self.__get_z(x,y)

    self.__add_v_to_zone(z1, vnum)
    self.VZ[vnum] = z1

    cdef long* new_vz

    if self.vnum>=self.vsize-1:

      new_vz = <long *>realloc(self.VZ, self.vsize*2*sizeof(long))

      if new_vz:
        self.VZ = new_vz;
        self.vsize = self.vsize*2
      else:
        ## this is really, really, bad
        return -1

    self.vnum += 1
    return vnum

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __del_vertex(self, long v1) nogil:

    self.__remove_v_from_zone(self.VZ[v1], v1)
    self.VZ[v1] = -1

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __add_v_to_zone(self, long z1, long v1) nogil:

    cdef sZ *z = self.Z[z1]

    z.ZV[z.count] = v1
    z.count += 1

    if z.count>=z.size-1:
      return self.__extend_zv_of_zone(z)

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __extend_zv_of_zone(self, sZ *z) nogil:

    cdef long new_size = z.size*2
    cdef long* new_zv = <long *>realloc(z.ZV, new_size*sizeof(long))

    if new_zv:
      z.ZV = new_zv;
      z.size = new_size
      if new_size>self.greatest_zone_size:
        self.greatest_zone_size = new_size
    else:
      ## this is really, really, bad
      return -1

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __remove_v_from_zone(self, long z1, long v1) nogil:

    cdef sZ *z = self.Z[z1]
    cdef long i

    for i in xrange(z.count):

      if z.ZV[i] == v1:
        z.ZV[i] = z.ZV[z.count-1]
        z.count -= 1
        return 1

    return -1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __get_z(self, double x, double y) nogil:

    cdef long nz = self.nz

    cdef long i = <long>(x*nz)
    cdef long j = <long>(y*nz)
    cdef long z = nz*i + j

    return z

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __update_v(self, long v1) nogil:

    cdef double x = self.X[v1]
    cdef double y = self.Y[v1]
    cdef long new_z = self.__get_z(x, y)
    cdef long old_z = self.VZ[v1]

    if old_z<0:
      return -1

    if new_z != old_z:

      self.__remove_v_from_zone(old_z, v1)
      self.__add_v_to_zone(new_z, v1)
      self.VZ[v1] = new_z

      return 1

    return -1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __sphere_is_free(self, double x, double y, double rad) nogil:
    """
    tests if there is another vertex within rad of x,y. rad must be less than
    the width of each zone.
    """

    cdef long a
    cdef long b
    cdef long i

    cdef long nz = self.nz

    cdef long zx = <long>(x*nz)
    cdef long zy = <long>(y*nz)

    cdef sZ *z

    cdef double dx
    cdef double dy
    cdef double rad2 = rad*rad

    # TOOD: check middle zone first
    for a in xrange(max(zx-1,0),min(zx+2,nz)):
      for b in xrange(max(zy-1,0),min(zy+2,nz)):

        z = self.Z[a*nz+b]

        for i in xrange(z.count):

          dx = x-self.X[z.ZV[i]]
          dy = y-self.Y[z.ZV[i]]

          if dx*dx+dy*dy<rad2:
            return -1

    return 1

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  @cython.cdivision(True)
  cdef long __sphere_vertices(self, double x, double y, double rad, long *vertices) nogil:

    cdef long a
    cdef long b
    cdef long i

    cdef long nz = self.nz

    cdef long zx = <long>(x*nz)
    cdef long zy = <long>(y*nz)

    cdef sZ *z

    cdef long num = 0

    cdef double dx
    cdef double dy
    cdef double rad2 = rad*rad

    for a in xrange(max(zx-1,0),min(zx+2,nz)):
      for b in xrange(max(zy-1,0),min(zy+2,nz)):

        z = self.Z[a*nz+b]

        for i in xrange(z.count):

          dx = x-self.X[z.ZV[i]]
          dy = y-self.Y[z.ZV[i]]

          if dx*dx+dy*dy<rad2:

            vertices[num] = z.ZV[i]
            num += 1

    return num

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __get_encode_zonemap_max_size(self) nogil:

    return self.total_zones * (3 + self.greatest_zone_size)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef void __encode_zonemap(self, long *a) nogil:

    cdef long z
    cdef long s
    cdef sZ *zone

    a[0] = self.vnum
    a[1] = self.nz
    a[2] = self.total_zones
    a[3] = self.greatest_zone_size

    cdef long i = 4
    for z in xrange(self.total_zones):

      zone = self.Z[z]
      a[i] = zone.count

      i += 1
      for s in xrange(zone.count):
        a[i] = zone.ZV[s]
        i += 1

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef void __decode_zonemap(self, long *a) nogil:

    cdef long* new_vz
    cdef long* new_zv

    cdef long vnum = a[0]
    cdef long nz = a[1]
    cdef long total_zones = a[2]
    cdef long greatest_zone_size = a[3]

    #TODO: VZ size
    if vnum>=self.vsize-1:
      new_vz = <long *>realloc(self.VZ, self.vsize*2*sizeof(long))
      if new_vz:
        self.VZ = new_vz;
        self.vsize = self.vsize*2
      else:
        ## this is really bad
        pass

    self.vnum = vnum
    self.nz = nz
    self.total_zones = total_zones
    self.greatest_zone_size = greatest_zone_size

    cdef long count = a[4]
    cdef long z = 0
    cdef long s = 0
    cdef long i = 5
    cdef long k = 0

    while True:

      if count>=self.Z[z].size-1:
        new_zv = <long *>realloc(self.Z[z].ZV, self.Z[z].size*2*sizeof(long))
        if new_zv:
          self.Z[z].ZV = new_zv;
          self.Z[z].size = self.Z[z].size*2
        else:
          ## this is really bad
          pass

      for s in xrange(count):

        #TODO: ZV sizes
        self.Z[z].ZV[s] = a[i]
        self.VZ[k] = z
        i += 1
        k += 1

      self.Z[z].count = count

      count = a[i]
      z += 1
      i += 1

      if z>=self.total_zones:
        break

    return

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long add_vertex(self, long v1):

    return self.__add_vertex(v1)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long del_vertex(self, long v1):

    return self.__del_vertex(v1)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cdef long __get_max_sphere_count(self) nogil:

    return self.greatest_zone_size*9

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef list _perftest(self, long nmax, long num_points):

    from time import time

    cdef np.ndarray[double, mode="c",ndim=2] a
    cdef long i
    cdef long num
    cdef double t1
    cdef double t2
    cdef list res = []
    cdef long asize
    cdef long *vertices

    cdef double *X = <double *>malloc(nmax*sizeof(double))
    cdef double *Y = <double *>malloc(nmax*sizeof(double))
    self.__assign_xy_arrays(X,Y)

    a = np.random.random((num_points,2))

    t1 = time()
    with nogil:
      for i in xrange(num_points):
        X[i] = a[i,0]
        Y[i] = a[i,1]
        self.__add_vertex(i)
    t2 = time()
    res.append(('add',t2-t1))

    t1 = time()
    with nogil:
      for i in xrange(num_points):
        self.__sphere_is_free(X[i], Y[i], 1.0/<double>self.nz)
    t2 = time()
    res.append(('free',t2-t1))

    t1 = time()
    with nogil:
      asize = self.__get_max_sphere_count()
      vertices = <long *>malloc(asize*sizeof(long))
      for i in xrange(num_points):
        self.__sphere_vertices(
          X[i],
          Y[i],
          1.0/<double>(self.nz),
          vertices
        )
    t2 = time()
    res.append(('sphere',t2-t1))

    t1 = time()
    with nogil:
      for i in xrange(num_points):
        self.__del_vertex(i)
    t2 = time()
    res.append(('del',t2-t1))

    free(X)
    free(Y)
    free(vertices)

    return res

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long update_v(self, long v1):

    return self.__update_v(v1)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long sphere_is_free(self, double x, double y, double rad):

    return self.__sphere_is_free(x, y, rad)

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long get_max_sphere_count(self):

    return self.__get_max_sphere_count()

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef long get_vnum(self):

    return self.vnum

  @cython.wraparound(False)
  @cython.boundscheck(False)
  @cython.nonecheck(False)
  cpdef list get_zone_info_dicts(self):

    cdef list res = []
    cdef dict d
    cdef long i

    for i in xrange(self.total_zones):

      d = {
        'i': self.Z[i].i,
        'size': self.Z[i].size,
        'count': self.Z[i].count
      }

      res.append(d)

    return res

