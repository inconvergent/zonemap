#!/usr/bin/python3
# -*- coding: utf-8 -*-

from zonemap import Zonemap


def main():

  nmax = 10000000
  num_points = 200000
  nzs = [20, 30, 40, 60, 80, 100]

  for nz in nzs:

    rad = 1.0/nz
    print(('zones\t\t{:d}'.format(nz)))
    print(('rad\t\t{:2.4f}'.format(rad)))
    print(' ')

    ZM = Zonemap(nz)
    res = ZM._perftest(nmax, num_points)
    for expl, perf in res:
      print(('{:s}\t\t{:3.10f}'.format(expl, perf)))
    del(ZM)

    print('\n\n')

  print('closing')

  return

if __name__ == '__main__' :

    main()

