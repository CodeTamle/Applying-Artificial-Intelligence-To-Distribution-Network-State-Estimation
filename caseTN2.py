# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 39 bus New England system.
"""

from numpy import array

def caseTN():
    """Power flow data for 39 bus New England system.
    Please see L{caseformat} for details on the case file format.

    Data taken from [1] with the following modifications/additions:

        - renumbered gen buses consecutively (as in [2] and [4])
        - added C{Pmin = 0} for all gens
        - added C{Qmin}, C{Qmax} for gens at 31 & 39 (copied from gen at 35)
        - added C{Vg} based on C{V} in bus data (missing for bus 39)
        - added C{Vg, Pg, Pd, Qd} at bus 39 from [2] (same in [4])
        - added C{Pmax} at bus 39: C{Pmax = Pg + 500}
        - added line flow limits and area data from [4]
        - added voltage limits, C{Vmax = 1.06, Vmin = 0.94}
        - added identical quadratic generator costs
        - increased C{Pmax} for gen at bus 34 from 308 to 508
          (assumed typo in [1], makes initial solved case feasible)
        - re-solved power flow

    Notes:
        - Bus 39, its generator and 2 connecting lines were added
          (by authors of [1]) to represent the interconnection with
          the rest of the eastern interconnect, and did not include
          C{Vg, Pg, Qg, Pd, Qd, Pmin, Pmax, Qmin} or C{Qmax}.
        - As the swing bus, bus 31 did not include and Q limits.
        - The voltages, etc in [1] appear to be quite close to the
          power flow solution of the case before adding bus 39 with
          it's generator and connecting branches, though the solution
          is not exact.
        - Explicit voltage setpoints for gen buses are not given, so
          they are taken from the bus data, however this results in two
          binding Q limits at buses 34 & 37, so the corresponding
          voltages have probably deviated from their original setpoints.
        - The generator locations and types are as follows:
            - 1   30      hydro
            - 2   31      nuke01
            - 3   32      nuke02
            - 4   33      fossil02
            - 5   34      fossil01
            - 6   35      nuke03
            - 7   36      fossil04
            - 8   37      nuke04
            - 9   38      nuke05
            - 10  39      interconnection to rest of US/Canada

    This is a solved power flow case, but it includes the following
    violations:
        - C{Pmax} violated at bus 31: C{Pg = 677.87, Pmax = 646}
        - C{Qmin} violated at bus 37: C{Qg = -1.37,  Qmin = 0}

    References:

    [1] G. W. Bills, et.al., I{"On-Line Stability Analysis Study"}
    RP90-1 Report for the Edison Electric Institute, October 12, 1970,
    pp. 1-20 - 1-35.
    prepared by
      - E. M. Gulachenski - New England Electric System
      - J. M. Undrill     - General Electric Co.
    "...generally representative of the New England 115 KV system, but is
    not an exact or complete model of any past, present or projected
    configuration of the actual New England 115 KV system."

    [2] M. A. Pai, I{Energy Function Analysis for Power System Stability},
    Kluwer Academic Publishers, Boston, 1989.
    (references [3] as source of data)

    [3] Athay, T.; Podmore, R.; Virmani, S., I{"A Practical Method for the
    Direct Analysis of Transient Stability,"} IEEE Transactions on Power
    Apparatus and Systems , vol.PAS-98, no.2, pp.573-584, March 1979.
    U{http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4113518&isnumber=4113486}
    (references [1] as source of data)

    [4] Data included with TC Calculator at
    U{http://www.pserc.cornell.edu/tcc/} for 39-bus system.

    @return: Power flow data for 39 bus New England system.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 500.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 19.60, 0.9, 0, 0, 1, 1, 0, 115, 1, 1.06, 0.94],
        [2, 2, 0, 0, 0, 0, 1, 0.9838, -5.78, 115, 1, 1.06, 0.94],
        [3, 1, 0, 0, 0, 0, 1, 0.97532, -6.64, 115, 1, 1.06, 0.94],
        [4, 1, 0, 0, 0, 0, 1, 0.97532, -6.64, 115, 1, 1.06, 0.94],
        [5, 1, 0, 0, 0, 0, 1, 0.97532, -6.64, 115, 1, 1.06, 0.94],
        [6, 1, 0, 0, 0, 0, 1, 0.97532, -6.64, 115, 1, 1.06, 0.94],
        [7, 2, 20.3, 0, 0, 7.07, 1, 0.99106, -8.77, 23, 1, 1.06, 0.94],
        [8, 1, 31, 4.7, 0, 0, 1, 0.95237, -12.14, 23, 1, 1.06, 0.94],
        [9, 1, 0, 0, 0, 0, 1, 0.98041, -1.85, 115, 1, 1.06, 0.94],
        [10, 1, 0, 0, 0, 0, 1, 0.98041, -1.85, 115, 1, 1.06, 0.94],
        [11, 1, 0, 0, 0, 0, 1, 0.97881, -1.5, 115, 1, 1.06, 0.94],
        [12, 1, 0, 0, 0, 0, 1, 0.9996, -0.01, 115, 1, 1.06, 0.94],
        [13, 1, 0, 0, 0, 0, 1, 0.98417, -1.52, 115, 1, 1.06, 0.94],
        [14, 1, 0, 0, 0, 0, 1, 0.99252, -0.64, 115, 1, 1.06, 0.94],
        [15, 1, 0, 0, 0, 0, 1, 1, -0.01, 115, 1, 1.06, 0.94],
        [16, 1, 0, 0, 0, 0, 1, 0.99059, -1.86, 115, 1, 1.06, 0.94],
        [17, 1, 25, 3, 0, 0, 1, 0.99002, -1.93, 115, 1, 1.06, 0.94],
        [18, 1, 158, 30, 0, 0, 1, 0.9998, -0.01, 115, 1, 1.06, 0.94],
        [19, 1, 0, 0, 0, 0, 1, 0.99708, -0.28, 115, 1, 1.06, 0.94],
        [20, 1, 0, 0, 0, 0, 1, 0.99708, -0.28, 115, 1, 1.06, 0.94],
        [21, 1, 0, 0, 0, 0, 1, 0.99504, -0.52, 115, 1, 1.06, 0.94],
        [22, 2, 18, 0, 0, 0, 1,  0.99504, -0.52, 115, 1, 1.06, 0.94],
        [23, 1, 20.3, 5.5, 0, 0, 1, 0.97426, -3.95, 23, 1, 1.06, 0.94],
        [24, 1, 0, 0, 0, 0, 1, 0.96545, -7.66, 115, 1, 1.06, 0.94],
        [25, 1, 0, 0, 0, 0, 1, 0.96596, -10.02, 115, 1, 1.06, 0.94],
        [26, 1, 0, 0, 0, 28, 1, 0.96596, -10.02, 115, 1, 1.06, 0.94],
        [27, 2, 0, 0, 0, 0, 1, 0.97143, -8.5, 115, 1, 1.06, 0.94],
        [28, 1, 0, 0, 0, 0, 1, 0.97143, -8.5, 115, 1, 1.06, 0.94],
        [29, 1, 0, 0, 0, 0, 1, 0.97309, -7.1, 110, 1, 1.06, 0.94],
        [30, 1, 0, 0, 0, 0, 1, 0.95480, -10.77, 115, 1, 1.06, 0.94],
        [31, 2, 17.5, 1.2, 0, 0, 1, 0.98242, -5.83, 115, 1, 1.06, 0.94],
        [32, 1, 0, 0, 0, 0, 1, 0.97529, -6.63, 115, 1, 1.06, 0.94],
        [33, 1, 19.1, 5.1, 0, 0, 1, 0.95069, -10.31, 115, 1, 1.06, 0.94],
        [34, 1, 12.1, 1.1, 0, 0, 1, 0.95623, -13.24, 115, 1, 1.06, 0.94],
        [35, 1, 18.6, 4.4, 0, 0, 1, 0.94061, -14.77, 115, 1, 1.06, 0.94],
        [36, 1, 26.4, 7, 0, 0, 1, 0.92695, -15.64, 23, 1, 1.06, 0.94],
        [37, 1, 13.1, 2.3, 0, 0, 1, 0.96665, -5.15, 115, 1, 1.06, 0.94],
        [38, 1, 15.9, 1.6, 0, 0, 1, 0.96846, -5.86, 115, 1, 1.06, 0.94],
        [39, 1, 30.6, 8.4, 0, 0, 1, 0.94451, -6.93, 115, 1, 1.06, 0.94],
        [40, 1, 10.1, 1.1, 0, 0, 1, 0.99213, -1.96, 115, 1, 1.06, 0.94],
        [41, 1, 15.3, 5.6, 0, 0, 1, 0.96815, -4.04, 115, 1, 1.06, 0.94]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 245.71, 28.99, 49500, -49500, 1, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 18.7, 2.3, 49500, -49500, 0.9838, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 0, 2.8, 49500, -49500, 0.99106, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [22, 0, 4.2, 49500, -49500,  0.99504, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [27, 20, 3, 49500, -49500, 0.97143, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [31, 34.8, 2.4, 49500, -49500, 0.98242, 500, 1, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [2, 4, 0.16601, 0.41019, 0.0018, 101, 0, 0, 0, 0, 1, -360, 360],
        [3, 5, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [4, 6, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [5, 6, 0, 0.00001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [5, 29, 0.02559, 0.06321, 0.0003, 185, 0, 0, 0, 0, 1, -360, 360],
        [5, 7, 0.0239, 0.8733, 0, 63, 0, 0, 0, 0, 1, -360, 360],
        [6, 8, 0.06425, 1.446, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [7, 8, 0, 0.1, 0, 600, 600, 600, 0, 0, 0, -360, 360],
        [9, 37, 0.0796, 2.097, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [10, 9, 0, 0.00001, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [10, 38, 0.07968, 2.097, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [11, 39, 0.0645, 1.4464, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [12, 1, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [12, 14, 0.03207, 0.10011, 0.0005, 121, 0, 0, 0, 0, 1, -360, 360],
        [13, 10, 0.04244, 0.10486, 0.0005, 101, 0, 0, 0, 0, 1, -360, 360],
        [14, 11, 0.11309, 0.27941, 0.0012, 101, 0, 0, 0, 0, 1, -360, 360],
        [14, 13, 0.0875, 0.27314, 0.0012, 121, 0, 0, 0, 0, 1, -360, 360],
        [15, 16, 0.05019, 0.15665, 0.0007, 121, 0, 0, 0, 0, 1, -360, 360],
        [15, 1, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [16, 17, 0.00833, 0.02614, 0.0001, 121, 0, 0, 0, 0, 1, -360, 360],
        [16, 32, 0.16628, 0.51912, 0.0023, 121, 0, 0, 0, 0, 1, -360, 360],
        [18, 19, 0.01588, 0.03924, 0.0002, 101, 0, 0, 0, 0, 1, -360, 360],
        [18, 1, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [19, 20, 0, 0.00001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [19, 40, 0.0645, 1.4464, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [20, 21, 0.02249, 0.05558, 0.0002, 101, 0, 0, 0, 0, 1, -360, 360],
        [20, 41, 0.0796, 2.0975, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [21, 22, 0, 0.00001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [21, 23, 0.06425, 1.446, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [24, 33, 0.041, 1.125, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [25, 26, 0, 0.00001, 0, 171.3, 0, 0, 0, 0, 1, -360, 360],
        [25, 34, 0.1272, 2.1617, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [26, 30, 0.10732, 0.26517, 0.0012, 101, 0, 0, 0, 0, 1, -360, 360],
        [26, 35, 0.07455, 2.04, 0, 25, 0, 0, 0, 0, 1, -360, 360],
        [27, 28, 0, 0.00001, 0, 171, 0, 0, 0, 0, 1, -360, 360],
        [28, 25, 0.08126, 0.20077, 0.0009, 101, 0, 0, 0, 0, 1, -360, 360],
        [29, 24, 0.10927, 0.26997, 0.001, 101, 0, 0, 0, 0, 1, -360, 360],
        [29, 27, 0.10753, 0.26568, 0.001, 101, 0, 0, 0, 0, 1, -360, 360],
        [30, 36, 0.0468, 1.438, 0, 40, 0, 0, 0, 0, 1, -360, 360],
        [31, 3, 0.16563, 0.40927, 0.0018, 101, 0, 0, 0, 0, 1, -360, 360],
        [32, 6, 0, 0.001, 0, 171, 0, 0, 0, 0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2],
        [2, 0, 0, 3, 0.01, 0.3, 0.2]
    ])

    return ppc
