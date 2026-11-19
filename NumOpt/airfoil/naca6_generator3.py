#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Airfoil Architecture & NACA 6-Series / 6A-Series Generator
====================================================================

A complete, object-oriented Python architecture centered around the `Airfoil` class
for storing, normalizing, modifying trailing-edge thickness, importing/exporting
(`.dat`, `.csv`, `np.ndarray`), and generating NACA 6-series and 6A-series airfoils.

Core Architecture
-----------------
1. `Airfoil` Class:
   - Stores upper and lower surface coordinates (`xu`, `yu`, `xl`, `yl`), continuous
     Selig contour coordinates (`coords`), camber/thickness distributions (`x`, `yc`, `yt`),
     and metadata (`name`, `le_radius`, `spec`).
   - `Airfoil.from_dat(filepath)`: Classmethod to construct an `Airfoil` from any
     Selig or Lednicer `.dat` file.
   - `Airfoil.from_numpy(coords, name=...)`: Classmethod to construct an `Airfoil`
     from a 2D NumPy array `(N, 2)` or `(2, N)`, a 2-tuple `(x, y)`, or a 4-tuple
     `(xu, yu, xl, yl)`.
   - `airfoil.with_te_thickness(te_thickness, ...)`: Returns a new `Airfoil` with
     modified trailing-edge thickness (supports both power-law wedge and XFOIL
     `TGAP` exponential blending).
   - `airfoil.normalize(derotate=True)` / `Airfoil.normalize(airfoil)`: Normalizes
     any airfoil into the unit chord interval `[0, 1]` with leading edge at `(0, 0)`
     and trailing edge at `x = 1.0`, returning a new `Airfoil`.
2. `generate_naca6(...) -> Airfoil`:
   - Generates any NACA 6-series or 6A-series airfoil (including decimal design lift
     coefficients like `"64-(1.5)12"`, custom `a`, and finite trailing-edge thickness)
     and returns an `Airfoil` instance.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

# =============================================================================
# 1. OFFICIAL NACA 6- & 6A-SERIES BASIC THICKNESS TABLES (IN % CHORD)
# =============================================================================

STANDARD_X_PERCENT: List[float] = [0.0, 0.5, 0.75, 1.25, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0]

# Family -> thickness (% chord) -> {"y": [26 ordinates in % chord], "le_radius": % chord}
THICKNESS_DATABASE: Dict[str, Dict[int, Dict[str, Union[List[float], float]]]] = {
    "63": {
        6: {"le_radius": 0.2680, "y": [0.0000, 0.5039, 0.6093, 0.7709, 1.0568, 1.4626, 1.7666, 2.0107, 2.3872, 2.6572, 2.8424, 2.9558, 2.9990, 2.9712, 2.8764, 2.7218, 2.5161, 2.2679, 1.9844, 1.6727, 1.3433, 1.0075, 0.6806, 0.3820, 0.1371, -0.0000]},
        8: {"le_radius": 0.4710, "y": [0.0000, 0.6682, 0.8087, 1.0244, 1.4073, 1.9516, 2.3588, 2.6857, 3.1895, 3.5500, 3.7964, 3.9458, 3.9998, 3.9577, 3.8252, 3.6131, 3.3336, 2.9982, 2.6172, 2.2003, 1.7616, 1.3171, 0.8871, 0.4963, 0.1777, 0.0000]},
        9: {"le_radius": 0.5900, "y": [0.0000, 0.7490, 0.9070, 1.1497, 1.5815, 2.1958, 2.6550, 3.0237, 3.5915, 3.9974, 4.2743, 4.4414, 4.5001, 4.4499, 4.2974, 4.0556, 3.7383, 3.3586, 2.9282, 2.4585, 1.9654, 1.4673, 0.9867, 0.5512, 0.1971, 0.0000]},
        10: {"le_radius": 0.7190, "y": [0.0000, 0.8286, 1.0040, 1.2739, 1.7548, 2.4396, 2.9512, 3.3619, 3.9941, 4.4454, 4.7529, 4.9374, 5.0003, 4.9414, 4.7682, 4.4959, 4.1401, 3.7158, 3.2357, 2.7131, 2.1659, 1.6146, 1.0841, 0.6047, 0.2160, -0.0000]},
        12: {"le_radius": 1.0040, "y": [0.0000, 0.9838, 1.1938, 1.5181, 2.0979, 2.9256, 3.5432, 4.0386, 4.8003, 5.3431, 5.7115, 5.9304, 6.0005, 5.9220, 5.7051, 5.3698, 4.9354, 4.4206, 3.8398, 3.2112, 2.5567, 1.9002, 1.2724, 0.7076, 0.2521, -0.0000]},
        15: {"le_radius": 1.4720, "y": [0.0000, 1.2039, 1.4654, 1.8712, 2.6018, 3.6488, 4.4285, 5.0532, 6.0121, 6.6934, 7.1533, 7.4222, 7.4992, 7.3860, 7.0988, 6.6640, 6.1072, 5.4529, 4.7194, 3.9318, 3.1176, 2.3073, 1.5389, 0.8524, 0.3026, 0.0000]},
        18: {"le_radius": 1.9510, "y": [0.0000, 1.4050, 1.7166, 2.2041, 3.0883, 4.3620, 5.3086, 6.0657, 7.2259, 8.0478, 8.5992, 8.9162, 8.9961, 8.8419, 8.4783, 7.9382, 7.2546, 6.4567, 5.5689, 4.6226, 3.6506, 2.6909, 1.7880, 0.9866, 0.3492, 0.0000]},
        21: {"le_radius": 2.4010, "y": [0.0000, 1.5829, 1.9430, 2.5116, 3.5525, 5.0615, 6.1810, 7.0744, 8.4410, 9.4059, 10.0493, 10.4129, 10.4911, 10.2906, 9.8438, 9.1921, 8.3775, 7.4333, 6.3890, 5.2835, 4.1572, 3.0531, 2.0217, 1.1116, 0.3924, -0.0000]},
    },
    "63A": {
        6: {"le_radius": 0.2780, "y": [0.0000, 0.5066, 0.6082, 0.7663, 1.0534, 1.4536, 1.7530, 1.9944, 2.3664, 2.6343, 2.8224, 2.9433, 2.9956, 2.9838, 2.9103, 2.7816, 2.6048, 2.3870, 2.1341, 1.8528, 1.5523, 1.2455, 0.9382, 0.6308, 0.3187, 0.0000]},
        8: {"le_radius": 0.4830, "y": [0.0000, 0.6687, 0.8037, 1.0152, 1.4004, 1.9372, 2.3384, 2.6617, 3.1595, 3.5175, 3.7684, 3.9286, 3.9957, 3.9766, 3.8745, 3.6988, 3.4594, 3.1662, 2.8271, 2.4509, 2.0514, 1.6454, 1.2391, 0.8325, 0.4216, 0.0000]},
        10: {"le_radius": 0.7250, "y": [0.0000, 0.8245, 0.9925, 1.2577, 1.7425, 2.4184, 2.9228, 3.3290, 3.9537, 4.4025, 4.7162, 4.9154, 4.9963, 4.9683, 4.8355, 4.6109, 4.3073, 3.9371, 3.5109, 3.0403, 2.5423, 2.0387, 1.5347, 1.0302, 0.5233, 0.0000]},
        12: {"le_radius": 0.9910, "y": [0.0000, 0.9722, 1.1725, 1.4919, 2.0782, 2.8957, 3.5051, 3.9953, 4.7484, 5.2887, 5.6657, 5.9032, 5.9971, 5.9588, 5.7934, 5.5183, 5.1489, 4.7004, 4.1862, 3.6212, 3.0256, 2.4255, 1.8254, 1.2240, 0.6238, 0.0000]},
        15: {"le_radius": 1.4030, "y": [0.0000, 1.1754, 1.4221, 1.8227, 2.5656, 3.6018, 4.3723, 4.9912, 5.9404, 6.6197, 7.0924, 7.3867, 7.4983, 7.4417, 7.2244, 6.8703, 6.3998, 5.8320, 5.1845, 4.4772, 3.7376, 2.9946, 2.2529, 1.5083, 0.7730, 0.0000]},
    },
    "64": {
        6: {"le_radius": 0.2490, "y": [0.0000, 0.4919, 0.5954, 0.7517, 1.0216, 1.4022, 1.6897, 1.9237, 2.2931, 2.5675, 2.7676, 2.9020, 2.9761, 2.9855, 2.9133, 2.7685, 2.5688, 2.3244, 2.0441, 1.7359, 1.4078, 1.0697, 0.7351, 0.4226, 0.1568, 0.0000]},
        8: {"le_radius": 0.4470, "y": [0.0000, 0.6550, 0.7928, 1.0008, 1.3615, 1.8718, 2.2570, 2.5706, 3.0654, 3.4325, 3.6995, 3.8781, 3.9754, 3.9831, 3.8794, 3.6797, 3.4077, 3.0773, 2.7001, 2.2875, 1.8501, 1.4015, 0.9601, 0.5499, 0.2032, -0.0000]},
        9: {"le_radius": 0.5660, "y": [0.0000, 0.7360, 0.8907, 1.1245, 1.5306, 2.1064, 2.5408, 2.8946, 3.4525, 3.8661, 4.1668, 4.3673, 4.4758, 4.4817, 4.3608, 4.1326, 3.8234, 3.4492, 3.0232, 2.5581, 2.0661, 1.5628, 1.0689, 0.6112, 0.2255, -0.0000]},
        10: {"le_radius": 0.6990, "y": [0.0000, 0.8165, 0.9880, 1.2474, 1.6991, 2.3406, 2.8246, 3.2186, 3.8400, 4.3004, 4.6346, 4.8570, 4.9766, 4.9800, 4.8409, 4.5834, 4.2365, 3.8181, 3.3429, 2.8252, 2.2788, 1.7211, 1.1754, 0.6710, 0.2470, 0.0000]},
        12: {"le_radius": 1.0050, "y": [0.0000, 0.9755, 1.1802, 1.4904, 2.0331, 2.8075, 3.3914, 3.8668, 4.6159, 5.1704, 5.5721, 5.8381, 5.9789, 5.9752, 5.7972, 5.4788, 5.0545, 4.5462, 3.9721, 3.3488, 2.6938, 2.0289, 1.3815, 0.7862, 0.2883, -0.0000]},
        15: {"le_radius": 1.5570, "y": [0.0000, 1.2077, 1.4609, 1.8458, 2.5250, 3.5019, 4.2382, 4.8374, 5.7808, 6.4780, 6.9814, 7.3125, 7.4832, 7.4629, 7.2202, 6.8050, 6.2609, 5.6145, 4.8898, 4.1079, 3.2917, 2.4688, 1.6740, 0.9484, 0.3459, 0.0000]},
        18: {"le_radius": 2.2080, "y": [0.0000, 1.4300, 1.7297, 2.1872, 3.0026, 4.1862, 5.0784, 5.8041, 6.9453, 7.7873, 8.3933, 8.7891, 8.9873, 8.9432, 8.6277, 8.1107, 7.4421, 6.6541, 5.7770, 4.8364, 3.8609, 2.8841, 1.9477, 1.0988, 0.3988, 0.0000]},
        21: {"le_radius": 2.9410, "y": [0.0000, 1.6399, 1.9838, 2.5114, 3.4618, 4.8572, 5.9094, 6.7645, 8.1078, 9.0971, 9.8069, 10.2669, 10.4903, 10.4155, 10.0198, 9.3950, 8.5971, 7.6648, 6.6334, 5.5351, 4.4025, 3.2762, 2.2043, 1.2386, 0.4474, -0.0000]},
    },
    "64A": {
        6: {"le_radius": 0.2770, "y": [0.0000, 0.5015, 0.6018, 0.7543, 1.0267, 1.4041, 1.6870, 1.9208, 2.2864, 2.5592, 2.7592, 2.8967, 2.9767, 2.9978, 2.9439, 2.8245, 2.6522, 2.4371, 2.1857, 1.9034, 1.5997, 1.2850, 0.9678, 0.6512, 0.3290, 0.0000]},
        8: {"le_radius": 0.4850, "y": [0.0000, 0.6614, 0.7945, 0.9981, 1.3631, 1.8692, 2.2484, 2.5615, 3.0508, 3.4157, 3.6826, 3.8657, 3.9715, 3.9965, 3.9194, 3.7558, 3.5222, 3.2325, 2.8951, 2.5177, 2.1138, 1.6967, 1.2777, 0.8587, 0.4349, 0.0000]},
        10: {"le_radius": 0.7360, "y": [0.0000, 0.8148, 0.9802, 1.2351, 1.6938, 2.3307, 2.8075, 3.2010, 3.8152, 4.2731, 4.6073, 4.8361, 4.9672, 4.9946, 4.8920, 4.6822, 4.3855, 4.0198, 3.5954, 3.1228, 2.6192, 2.1010, 1.5817, 1.0620, 0.5393, 0.0000]},
        12: {"le_radius": 1.0210, "y": [0.0000, 0.9599, 1.1571, 1.4632, 2.0172, 2.7874, 3.3633, 3.8383, 4.5791, 5.1309, 5.5329, 5.8077, 5.9638, 5.9916, 5.8615, 5.6034, 5.2423, 4.7994, 4.2870, 3.7191, 3.1163, 2.4983, 1.8801, 1.2613, 0.6424, 0.0000]},
        15: {"le_radius": 1.4810, "y": [0.0000, 1.1586, 1.4015, 1.7844, 2.4845, 3.4601, 4.1883, 4.7882, 5.7223, 6.4174, 6.9224, 7.2668, 7.4597, 7.4852, 7.3100, 6.9761, 6.5154, 5.9546, 5.3091, 4.5981, 3.8475, 3.0824, 2.3189, 1.5532, 0.7951, 0.0000]},
    },
    "65": {
        6: {"le_radius": 0.2300, "y": [0.0000, 0.4680, 0.5639, 0.7052, 0.9520, 1.3110, 1.5905, 1.8243, 2.1977, 2.4814, 2.6957, 2.8510, 2.9505, 2.9971, 2.9819, 2.8978, 2.7413, 2.5198, 2.2475, 1.9345, 1.5921, 1.2311, 0.8633, 0.5069, 0.1892, -0.0000]},
        8: {"le_radius": 0.4060, "y": [0.0000, 0.6205, 0.7477, 0.9355, 1.2653, 1.7464, 2.1204, 2.4333, 2.9327, 3.3120, 3.5981, 3.8050, 3.9370, 3.9975, 3.9733, 3.8556, 3.6401, 3.3388, 2.9714, 2.5508, 2.0943, 1.6145, 1.1286, 0.6602, 0.2463, 0.0000]},
        9: {"le_radius": 0.5100, "y": [0.0000, 0.6955, 0.8383, 1.0492, 1.4206, 1.9632, 2.3850, 2.7377, 3.3004, 3.7276, 4.0499, 4.2826, 4.4307, 4.4978, 4.4684, 4.3329, 4.0865, 3.7443, 3.3284, 2.8539, 2.3402, 1.8014, 1.2573, 0.7342, 0.2739, 0.0000]},
        10: {"le_radius": 0.6240, "y": [0.0000, 0.7697, 0.9278, 1.1616, 1.5748, 2.1794, 2.6491, 3.0418, 3.6681, 4.1435, 4.5020, 4.7605, 4.9247, 4.9981, 4.9631, 4.8089, 4.5309, 4.1471, 3.6823, 3.1537, 2.5827, 1.9851, 1.3834, 0.8064, 0.3010, 0.0000]},
        12: {"le_radius": 0.8760, "y": [0.0000, 0.9146, 1.1031, 1.3824, 1.8792, 2.6091, 3.1756, 3.6490, 4.4035, 4.9758, 5.4070, 5.7172, 5.9135, 5.9990, 5.9512, 5.7573, 5.4137, 4.9445, 4.3806, 3.7427, 3.0576, 2.3433, 1.6282, 0.9461, 0.3534, 0.0000]},
        15: {"le_radius": 1.3010, "y": [0.0000, 1.1222, 1.3548, 1.7012, 2.3240, 3.2454, 3.9598, 4.5562, 5.5056, 6.2249, 6.7662, 7.1543, 7.3985, 7.5004, 7.4295, 7.1704, 6.7225, 6.1200, 5.4045, 4.6011, 3.7453, 2.8588, 1.9777, 1.1444, 0.4281, 0.0000]},
        18: {"le_radius": 1.7560, "y": [0.0000, 1.3152, 1.5900, 2.0018, 2.7507, 3.8687, 4.7349, 5.4570, 6.6050, 7.4738, 8.1267, 8.5934, 8.8854, 9.0016, 8.9025, 8.5722, 8.0118, 7.2713, 6.4011, 5.4316, 4.4047, 3.3493, 2.3072, 1.3305, 0.4987, 0.0000]},
        21: {"le_radius": 2.2120, "y": [0.0000, 1.4909, 1.8058, 2.2805, 3.1550, 4.4755, 5.4980, 6.3493, 7.7005, 8.7216, 9.4880, 10.0344, 10.3740, 10.5025, 10.3711, 9.9609, 9.2823, 8.3992, 7.3699, 6.2331, 5.0374, 3.8165, 2.6181, 1.5056, 0.5656, 0.0000]},
    },
    "65A": {
        6: {"le_radius": 0.2530, "y": [0.0000, 0.4785, 0.5757, 0.7279, 0.9890, 1.3199, 1.5958, 1.8282, 2.1962, 2.4754, 2.6872, 2.8415, 2.9439, 2.9949, 2.9908, 2.9247, 2.7931, 2.6014, 2.3631, 2.0859, 1.7752, 1.4389, 1.0858, 0.7307, 0.3693, 0.0000]},
        8: {"le_radius": 0.4370, "y": [0.0000, 0.6303, 0.7597, 0.9633, 1.3124, 1.7556, 2.1256, 2.4365, 2.9289, 3.3022, 3.5851, 3.7910, 3.9273, 3.9942, 3.9866, 3.8946, 3.7143, 3.4545, 3.1338, 2.7623, 2.3473, 1.8997, 1.4320, 0.9626, 0.4873, 0.0000]},
        10: {"le_radius": 0.6540, "y": [0.0000, 0.7753, 0.9368, 1.1923, 1.6303, 2.1872, 2.6525, 3.0429, 3.6607, 4.1288, 4.4834, 4.7413, 4.9114, 4.9938, 4.9816, 4.8618, 4.6306, 4.3010, 3.8966, 3.4298, 2.9104, 2.3518, 1.7714, 1.1893, 0.6033, 0.0000]},
        12: {"le_radius": 0.8900, "y": [0.0000, 0.9120, 1.1052, 1.4133, 1.9415, 2.6134, 3.1755, 3.6465, 4.3911, 4.9550, 5.3819, 5.6920, 5.8961, 5.9937, 5.9758, 5.8264, 5.5421, 5.1411, 4.6518, 4.0889, 3.4650, 2.7959, 2.1043, 1.4110, 0.7173, 0.0000]},
        15: {"le_radius": 1.2530, "y": [0.0000, 1.0979, 1.3375, 1.7256, 2.3921, 3.2399, 3.9501, 4.5444, 5.4826, 6.1922, 6.7291, 7.1185, 7.3738, 7.4938, 7.4655, 7.2686, 6.9009, 6.3898, 5.7711, 5.0631, 4.2819, 3.4477, 2.5920, 1.7352, 0.8854, 0.0000]},
    },
    "66": {
        6: {"le_radius": 0.2180, "y": [0.0000, 0.4592, 0.5529, 0.6896, 0.9174, 1.2562, 1.5255, 1.7511, 2.1187, 2.3999, 2.6171, 2.7810, 2.8973, 2.9697, 2.9986, 2.9839, 2.9229, 2.8083, 2.6116, 2.3136, 1.9511, 1.5436, 1.1066, 0.6681, 0.2625, 0.0000]},
        8: {"le_radius": 0.3900, "y": [0.0000, 0.6099, 0.7338, 0.9147, 1.2182, 1.6719, 2.0320, 2.3337, 2.8248, 3.2003, 3.4903, 3.7091, 3.8642, 3.9604, 3.9984, 3.9777, 3.8946, 3.7378, 3.4657, 3.0592, 2.5714, 2.0258, 1.4453, 0.8674, 0.3381, 0.0000]},
        9: {"le_radius": 0.4930, "y": [0.0000, 0.6844, 0.8231, 1.0259, 1.3672, 1.8787, 2.2845, 2.6244, 3.1776, 3.6004, 3.9270, 4.1733, 4.3478, 4.4558, 4.4983, 4.4745, 4.3799, 4.2012, 3.8895, 3.4272, 2.8760, 2.2610, 1.6093, 0.9631, 0.3739, -0.0000]},
        10: {"le_radius": 0.6080, "y": [0.0000, 0.7582, 0.9116, 1.1360, 1.5150, 2.0846, 2.5364, 2.9146, 3.5301, 4.0003, 4.3635, 4.6374, 4.8314, 4.9513, 4.9983, 4.9712, 4.8649, 4.6636, 4.3111, 3.7920, 3.1768, 2.4922, 1.7699, 1.0561, 0.4084, 0.0000]},
        12: {"le_radius": 0.8710, "y": [0.0000, 0.9039, 1.0858, 1.3525, 1.8067, 2.4936, 3.0378, 3.4933, 4.2340, 4.7996, 5.2364, 5.5656, 5.7987, 5.9424, 5.9981, 5.9642, 5.8339, 5.5857, 5.1477, 4.5122, 3.7677, 2.9432, 2.0807, 1.2346, 0.4738, 0.0000]},
        15: {"le_radius": 1.3450, "y": [0.0000, 1.1157, 1.3386, 1.6666, 2.2326, 3.0980, 3.7831, 4.3558, 5.2863, 5.9963, 6.5444, 6.9573, 7.2494, 7.4291, 7.4979, 7.4531, 7.2853, 6.9614, 6.3846, 5.5686, 4.6263, 3.5911, 2.5222, 1.4839, 0.5635, 0.0000]},
        18: {"le_radius": 1.9040, "y": [0.0000, 1.3179, 1.5791, 1.9652, 2.6413, 3.6886, 4.5176, 5.2097, 6.3330, 7.1892, 7.8502, 8.3477, 8.6995, 8.9159, 8.9977, 8.9415, 8.7343, 8.3274, 7.5989, 6.5958, 5.4522, 4.2058, 2.9351, 1.7130, 0.6441, -0.0000]},
        21: {"le_radius": 2.5370, "y": [0.0000, 1.5086, 1.8051, 2.2457, 3.0294, 4.2624, 5.2389, 6.0529, 7.3726, 8.3773, 9.1527, 9.7361, 10.1487, 10.4025, 10.4976, 10.4297, 10.1809, 9.6829, 8.7908, 7.5948, 6.2451, 4.7886, 3.3209, 1.9235, 0.7166, 0.0000]},
    },
    "67": {
        12: {"le_radius": 1.0120, "y": [0.0000, 0.9505, 1.1372, 1.4355, 1.9412, 2.6174, 3.1377, 3.5617, 4.2379, 4.7606, 5.1698, 5.4878, 5.7271, 5.8891, 5.9804, 5.9972, 5.9368, 5.7860, 5.5292, 5.1376, 4.5357, 3.6903, 2.7246, 1.6745, 0.6565, 0.0000]},
        15: {"le_radius": 1.5700, "y": [0.0000, 1.1649, 1.3922, 1.7626, 2.3941, 3.2443, 3.8995, 4.4329, 5.2829, 5.9398, 6.4539, 6.8539, 7.1550, 7.3593, 7.4750, 7.4969, 7.4214, 7.2309, 6.9052, 6.4023, 5.6213, 4.5401, 3.3271, 2.0210, 0.7880, 0.0000]},
    },
}


# =============================================================================
# 2. NACA 6-SERIES SPECIFICATION & DESIGNATION PARSER
# =============================================================================

@dataclass
class NACA6Spec:
    """Parsed specification for a NACA 6- or 6A-series airfoil."""
    family: str                 # '63', '63A', '64', '64A', '65', '65A', '66', '67', or 'CUSTOM'
    c_li: float                 # Design lift coefficient (e.g., 0.2 for -212, 0.15 for -(1.5)12)
    t_c: float                  # Maximum thickness-to-chord ratio (e.g., 0.12 for -212)
    a: float = 1.0              # Mean-line chordwise uniform-load fraction (0.0 to 1.0)
    modified_camber: bool = False  # True for 6A-series modified a=0.8 camber line
    low_drag_range: Optional[float] = None  # Half-width of low-drag bucket in tenths
    te_thickness: float = 0.0   # Total trailing-edge thickness t_TE/c (fraction of chord)
    designation: str = ""       # Compact designation without 'NACA' or spaces


def _format_cli_tenths(c_li: float) -> str:
    """Format 10 * c_li as a single digit (e.g. '2') or parenthesized value (e.g. '(1.5)')."""
    cli_tenths = round(c_li * 10.0, 6)
    if math.isclose(cli_tenths, round(cli_tenths), abs_tol=1e-6) and 0 <= int(round(cli_tenths)) <= 9:
        return str(int(round(cli_tenths)))
    return f"({cli_tenths:g})"


def _format_tc_percent(t_c: float) -> str:
    """Format 100 * t_c as 2 digits (e.g. '12') or parenthesized decimal (e.g. '(12.5)')."""
    tc_pct = round(t_c * 100.0, 6)
    if math.isclose(tc_pct, round(tc_pct), abs_tol=1e-6) and 0 <= int(round(tc_pct)) <= 99:
        return f"{int(round(tc_pct)):02d}"
    return f"({tc_pct:g})"


def parse_naca6_designation(designation: str) -> NACA6Spec:
    """
    Parse a compact NACA 6- or 6A-series designation string into a `NACA6Spec`.

    Supported examples (no 'NACA' prefix or spaces needed):
      - "63-212"
      - "64-(1.5)12" or "64(1)-(1.5)12" or "64A(1.5)12" (decimal design C_li = 0.15)
      - "64(1)-212" or "641-212" or "64,1-212"
      - "64(1)-212,a=0.6" or "65(3)-418,a=0.5"
      - "64A210" or "64A-210"
      - "66(2)-015"
    """
    raw = designation.strip()

    # Extract optional "a=..." parameter (no space required before/after comma)
    a_val: Optional[float] = None
    a_match = re.search(r"[,;\s]*a\s*=\s*([0-9]*\.?[0-9]+)", raw, flags=re.IGNORECASE)
    if a_match:
        a_val = float(a_match.group(1))
        raw = (raw[:a_match.start()] + raw[a_match.end():]).strip()

    # Strip optional leading "NACA" if present
    cleaned = re.sub(r"^NACA\s*", "", raw, flags=re.IGNORECASE).strip()

    pattern = re.compile(
        r"^(6[3-7])"                                         # Group 1: Family base (63..67)
        r"(?:(A)|(?:\((\d+(?:\.\d+)?)\)|,(\d+(?:\.\d+)?)|(\d))?)"  # Group 2..5: 'A' or low-drag subscript
        r"[-\s]?"                                           # Optional hyphen or space
        r"(?:\((\d+(?:\.\d+)?)\)|(\d+\.\d+)|(\d))"  # Group 6..8: C_li in tenths: (1.5), 1.5, or 2
        r"(?:\((\d+(?:\.\d+)?)\)|(\d{2}))$",           # Group 9..10: Thickness in % chord: (12.5) or 12
        flags=re.IGNORECASE
    )
    m = pattern.match(cleaned)
    if not m:
        raise ValueError(
            f"Unrecognized 6-series designation: '{designation}'. "
            "Examples: '64(1)-212', '64-(1.5)12', '65(3)-418,a=0.5', '64A210', '63-212'."
        )

    base_fam = m.group(1)
    is_a_series = m.group(2) is not None
    sub_str_raw = m.group(3) or m.group(4) or m.group(5)
    cli_tenths_str = m.group(6) or m.group(7) or m.group(8)
    tc_pct_str = m.group(9) or m.group(10)

    family = f"{base_fam}A" if is_a_series else base_fam
    c_li = round(float(cli_tenths_str) / 10.0, 10)
    t_c = round(float(tc_pct_str) / 100.0, 10)
    low_drag = round(float(sub_str_raw) / 10.0, 10) if sub_str_raw is not None else None

    if is_a_series:
        modified_camber = True if a_val is None else False
        a_final = 0.8 if a_val is None else a_val
    else:
        modified_camber = False
        a_final = 1.0 if a_val is None else a_val

    cli_fmt = _format_cli_tenths(c_li)
    tc_fmt = _format_tc_percent(t_c)

    if is_a_series:
        norm_name = f"{family}{cli_fmt}{tc_fmt}"
    else:
        sub_str = f"({low_drag * 10:g})" if low_drag is not None else ""
        a_str = f",a={a_final:g}" if not math.isclose(a_final, 1.0) and not math.isclose(c_li, 0.0) else ""
        norm_name = f"{base_fam}{sub_str}-{cli_fmt}{tc_fmt}{a_str}"

    return NACA6Spec(
        family=family,
        c_li=c_li,
        t_c=t_c,
        a=a_final,
        modified_camber=modified_camber,
        low_drag_range=low_drag,
        te_thickness=0.0,
        designation=norm_name,
    )


# =============================================================================
# 3. UNIVERSAL AIRFOIL CLASS (`Airfoil`)
# =============================================================================

class Airfoil:
    """
    Universal Airfoil class storing airfoil surface coordinates and providing
    classmethods for importing `.dat` files and NumPy arrays, normalizing to
    `[0, 1]`, and modifying trailing-edge thickness.

    Attributes
    ----------
    name : str
        Airfoil name or designation (e.g. `"64-(1.5)12"`, `"RAE2822"`).
    xu, yu : np.ndarray
        Upper surface coordinates ordered from leading edge (`x_LE`) to trailing edge (`x_TE`).
    xl, yl : np.ndarray
        Lower surface coordinates ordered from leading edge (`x_LE`) to trailing edge (`x_TE`).
    x : np.ndarray
        Chordwise reference stations from leading edge to trailing edge.
    yt : np.ndarray
        Half-thickness distribution `y_t` at stations `x`.
    yc : np.ndarray
        Mean camber line `y_c` at stations `x`.
    dyc_dx : np.ndarray
        Mean camber line slope `dy_c/dx` at stations `x`.
    theta : np.ndarray
        Camber line angle `arctan(dy_c/dx)` [radians].
    le_radius : float
        Leading-edge radius `r_LE` (in chord units).
    spec : NACA6Spec
        Metadata specification (populated automatically for both NACA 6-series and custom foils).
    """

    def __init__(
        self,
        xu: np.ndarray,
        yu: np.ndarray,
        xl: np.ndarray,
        yl: np.ndarray,
        *,
        name: str = "Airfoil",
        x: Optional[np.ndarray] = None,
        yt: Optional[np.ndarray] = None,
        yc: Optional[np.ndarray] = None,
        dyc_dx: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        le_radius: float = 0.0,
        spec: Optional[NACA6Spec] = None,
    ) -> None:
        self.xu = np.asarray(xu, dtype=float).copy()
        self.yu = np.asarray(yu, dtype=float).copy()
        self.xl = np.asarray(xl, dtype=float).copy()
        self.yl = np.asarray(yl, dtype=float).copy()
        self.name = str(name).strip() or "Airfoil"
        self.le_radius = float(le_radius)

        # Compute mean camber line (yc) and half-thickness (yt) if not explicitly provided
        if x is None or yt is None or yc is None:
            x_est, yt_est, yc_est = self._estimate_camber_and_thickness(
                self.xu, self.yu, self.xl, self.yl
            )
            self.x = x_est if x is None else np.asarray(x, dtype=float).copy()
            self.yt = yt_est if yt is None else np.asarray(yt, dtype=float).copy()
            self.yc = yc_est if yc is None else np.asarray(yc, dtype=float).copy()
        else:
            self.x = np.asarray(x, dtype=float).copy()
            self.yt = np.asarray(yt, dtype=float).copy()
            self.yc = np.asarray(yc, dtype=float).copy()

        if dyc_dx is None:
            self.dyc_dx = (
                np.gradient(self.yc, self.x, edge_order=1)
                if len(self.x) > 2
                else np.zeros_like(self.x)
            )
        else:
            self.dyc_dx = np.asarray(dyc_dx, dtype=float).copy()

        self.theta = np.arctan(self.dyc_dx) if theta is None else np.asarray(theta, dtype=float).copy()

        if spec is None:
            chord_val = self.chord
            t_c_est = float(2.0 * np.max(self.yt) / chord_val) if chord_val > 0 else 0.0
            te_est = float(max(0.0, self.yu[-1] - self.yl[-1]) / chord_val) if chord_val > 0 else 0.0
            self.spec = NACA6Spec(
                family="CUSTOM",
                c_li=0.0,
                t_c=t_c_est,
                a=1.0,
                modified_camber=False,
                te_thickness=te_est,
                designation=self.name,
            )
        else:
            self.spec = spec
            if not self.name or self.name == "Airfoil":
                self.name = spec.designation

    @staticmethod
    def _estimate_camber_and_thickness(
        xu: np.ndarray, yu: np.ndarray, xl: np.ndarray, yl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate chordwise stations `x`, half-thickness `yt`, and camber `yc` from upper/lower surfaces."""
        if len(xu) == len(xl) and np.allclose(xu, xl, atol=1e-2):
            x = 0.5 * (xu + xl)
            yc = 0.5 * (yu + yl)
            yt = np.maximum(0.5 * (yu - yl), 0.0)
        else:
            x_min = float(min(xu[0], xl[0]))
            x_max = float(max(xu[-1], xl[-1]))
            n_pts = max(len(xu), len(xl))
            beta = np.linspace(0.0, math.pi, n_pts)
            x = x_min + 0.5 * (x_max - x_min) * (1.0 - np.cos(beta))
            yu_i = np.interp(x, xu, yu)
            yl_i = np.interp(x, xl, yl)
            yc = 0.5 * (yu_i + yl_i)
            yt = np.maximum(0.5 * (yu_i - yl_i), 0.0)
        return x, yt, yc

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def chord(self) -> float:
        """Chord length along the x-axis (`max(x_TE) - min(x_LE)`)."""
        x_le = float(min(np.min(self.xu), np.min(self.xl)))
        x_te = float(max(np.max(self.xu), np.max(self.xl)))
        return max(x_te - x_le, 1e-12)

    @property
    def coords(self) -> np.ndarray:
        """
        Return `(N, 2)` NumPy array of continuous contour coordinates in Selig order:
        Upper TE -> LE -> Lower TE.
        """
        x_s, y_s = self.to_selig_coords()
        return np.column_stack([x_s, y_s])

    @property
    def actual_te_thickness(self) -> float:
        """Total trailing-edge thickness gap (`yu[-1] - yl[-1]`)."""
        return float(self.yu[-1] - self.yl[-1])

    @property
    def te_thickness(self) -> float:
        """Alias for `actual_te_thickness`."""
        return self.actual_te_thickness

    def __repr__(self) -> str:
        return (
            f"Airfoil(name='{self.name}', n_upper={len(self.xu)}, n_lower={len(self.xl)}, "
            f"chord={self.chord:.4f}, t_c={self.spec.t_c*100:.2f}%, te_gap={self.actual_te_thickness:.6f})"
        )

    # -------------------------------------------------------------------------
    # Classmethods: Construct Airfoil from .dat files or NumPy arrays
    # -------------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        coords: Union[
            np.ndarray,
            Sequence[Sequence[float]],
            Tuple[np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
        name: str = "Airfoil",
    ) -> "Airfoil":
        """
        Classmethod to construct an `Airfoil` from a NumPy array or tuple of arrays.

        Supported formats for `coords`:
          1. 2D NumPy array of shape `(N, 2)` or `(2, N)` in Selig contour order
             (Upper TE -> LE -> Lower TE).
          2. 2-tuple `(x, y)` of 1D arrays in Selig contour order.
          3. 4-tuple `(xu, yu, xl, yl)` of 1D surface arrays (LE -> TE).
        """
        if isinstance(coords, tuple):
            if len(coords) == 4:
                xu, yu, xl, yl = (np.asarray(a, dtype=float) for a in coords)
                if xu[0] > xu[-1]:
                    xu, yu = xu[::-1], yu[::-1]
                if xl[0] > xl[-1]:
                    xl, yl = xl[::-1], yl[::-1]
                return cls(xu=xu, yu=yu, xl=xl, yl=yl, name=name)
            elif len(coords) == 2:
                x_in, y_in = np.asarray(coords[0], dtype=float), np.asarray(coords[1], dtype=float)
                pts = np.column_stack([x_in, y_in])
            else:
                raise ValueError(
                    f"Tuple input to Airfoil.from_numpy must have length 2 or 4, got {len(coords)}."
                )
        else:
            arr = np.asarray(coords, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D coordinate array, got shape {arr.shape}.")
            if arr.shape[1] == 2:
                pts = arr
            elif arr.shape[0] == 2:
                pts = arr.T
            else:
                raise ValueError(f"Expected array shape (N, 2) or (2, N), got {arr.shape}.")

        # Split Selig loop (Upper TE -> LE -> Lower TE) at the leading edge
        # Using maximum distance from the trailing-edge midpoint is rotationally invariant
        te_mid = 0.5 * (pts[0] + pts[-1])
        dists = np.hypot(pts[:, 0] - te_mid[0], pts[:, 1] - te_mid[1])
        le_idx = int(np.argmax(dists))
        if le_idx == 0 or le_idx == len(pts) - 1:
            raise ValueError(
                "2D coordinate array must be ordered along the contour (Upper TE -> LE -> Lower TE) "
                "so the leading edge is an interior point."
            )

        upper_pts = pts[: le_idx + 1][::-1]  # LE -> Upper TE
        lower_pts = pts[le_idx:]             # LE -> Lower TE
        return cls(
            xu=upper_pts[:, 0],
            yu=upper_pts[:, 1],
            xl=lower_pts[:, 0],
            yl=lower_pts[:, 1],
            name=name,
        )

    @classmethod
    def from_dat(cls, filepath: Union[str, Path]) -> "Airfoil":
        """
        Classmethod to read any `.dat` airfoil file (Selig continuous format or
        Lednicer two-block format) and return an `Airfoil` instance.
        """
        coords, title = load_dat_coords(filepath)
        return cls.from_numpy(coords, name=title)

    # -------------------------------------------------------------------------
    # Transformation Methods (returning a new Airfoil instance)
    # -------------------------------------------------------------------------

    def normalize(self, derotate: bool = True) -> "Airfoil":
        """
        Normalize the airfoil coordinates into `[0, 1]` and return a new `Airfoil`:
          1. Locate the leading-edge apex (point furthest from the trailing-edge
             midpoint, which is rotationally invariant) and translate it to `(0.0, 0.0)`.
          2. If `derotate=True`, rotate the airfoil so the chord line connecting the
             leading edge and the trailing-edge midpoint lies along the positive x-axis.
          3. Shift `min(x)` to `0.0` and scale isotropically so `x` spans `[0.0, 1.0]`.
        """
        pts = self.coords.copy()
        te_mid = 0.5 * (pts[0] + pts[-1])
        dists = np.hypot(pts[:, 0] - te_mid[0], pts[:, 1] - te_mid[1])
        le_idx = int(np.argmax(dists))

        # 1. Translate rotation-invariant leading-edge apex to (0, 0)
        pts -= pts[le_idx]

        # 2. Derotate chord line if requested
        if derotate:
            te_mid = 0.5 * (pts[0] + pts[-1])
            angle = math.atan2(float(te_mid[1]), float(te_mid[0]))
            if not math.isclose(angle, 0.0, abs_tol=1e-14):
                ca, sa = math.cos(-angle), math.sin(-angle)
                pts = np.column_stack([pts[:, 0] * ca - pts[:, 1] * sa, pts[:, 0] * sa + pts[:, 1] * ca])

            le_idx = int(np.argmin(pts[:, 0]))
            pts -= pts[le_idx]
            te_mid = 0.5 * (pts[0] + pts[-1])
            angle2 = math.atan2(float(te_mid[1]), float(te_mid[0]))
            if not math.isclose(angle2, 0.0, abs_tol=1e-14):
                ca2, sa2 = math.cos(-angle2), math.sin(-angle2)
                pts = np.column_stack([pts[:, 0] * ca2 - pts[:, 1] * sa2, pts[:, 0] * sa2 + pts[:, 1] * ca2])

        # 3. Shift min(x) to 0.0 and scale max(x) to 1.0
        le_idx = int(np.argmin(pts[:, 0]))
        x_min = float(pts[le_idx, 0])
        pts[:, 0] -= x_min
        x_max = float(np.max(pts[:, 0]))
        if x_max <= 1e-12:
            raise ValueError("Cannot normalize degenerate airfoil with zero chord length.")

        pts /= x_max
        pts[le_idx, 0] = 0.0

        upper_pts = pts[: le_idx + 1][::-1]
        lower_pts = pts[le_idx:]

        new_spec = NACA6Spec(
            family=self.spec.family,
            c_li=self.spec.c_li,
            t_c=self.spec.t_c,
            a=self.spec.a,
            modified_camber=self.spec.modified_camber,
            low_drag_range=self.spec.low_drag_range,
            te_thickness=float(max(0.0, upper_pts[-1, 1] - lower_pts[-1, 1])),
            designation=self.spec.designation,
        )

        return Airfoil(
            xu=upper_pts[:, 0],
            yu=upper_pts[:, 1],
            xl=lower_pts[:, 0],
            yl=lower_pts[:, 1],
            name=self.name,
            le_radius=self.le_radius / x_max,
            spec=new_spec,
        )


    def with_te_thickness(
        self,
        te_thickness: float,
        te_blend_power: float = 1.0,
        blend_mode: Literal["power", "xfoil"] = "power",
        relative: bool = True,
    ) -> "Airfoil":
        """
        Return a new `Airfoil` instance with its trailing-edge thickness modified
        to `te_thickness`.

        Parameters
        ----------
        te_thickness : float
            Target total trailing-edge thickness (`y_u,TE - y_l,TE`). When
            `relative=True` (default), given as a fraction of chord `t_TE/c`
            (e.g. `0.002` for 0.2% chord).
        te_blend_power : float, default 1.0
            Chordwise blending exponent `p >= 1.0` for `(x/c)^p` when `blend_mode="power"`.
        blend_mode : {"power", "xfoil"}, default "power"
            Blending function shape (`"power"` or `"xfoil"`).
        relative : bool, default True
            Whether `te_thickness` is relative to chord length `c`.
        """
        xu_new, yu_new, xl_new, yl_new = _modify_upper_lower_arrays(
            self.xu,
            self.yu,
            self.xl,
            self.yl,
            te_thickness=te_thickness,
            te_blend_power=te_blend_power,
            blend_mode=blend_mode,
            relative=relative,
        )
        target_rel_te = te_thickness if relative else te_thickness / self.chord
        yt_new = apply_te_thickness(
            self.x,
            self.yt,
            te_thickness=target_rel_te * self.chord,
            te_blend_power=te_blend_power,
            blend_mode=blend_mode,
        )
        new_spec = NACA6Spec(
            family=self.spec.family,
            c_li=self.spec.c_li,
            t_c=self.spec.t_c,
            a=self.spec.a,
            modified_camber=self.spec.modified_camber,
            low_drag_range=self.spec.low_drag_range,
            te_thickness=float(target_rel_te),
            designation=self.spec.designation,
        )
        return Airfoil(
            xu=xu_new,
            yu=yu_new,
            xl=xl_new,
            yl=yl_new,
            name=self.name,
            x=self.x,
            yt=yt_new,
            yc=self.yc,
            dyc_dx=self.dyc_dx,
            theta=self.theta,
            le_radius=self.le_radius,
            spec=new_spec,
        )

    # -------------------------------------------------------------------------
    # Export & Visualization Methods
    # -------------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return `(N, 2)` NumPy array of Selig contour coordinates (Upper TE -> LE -> Lower TE)."""
        return self.coords

    def to_selig_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return continuous loop coordinates `(x, y)` in Selig / XFOIL order:
        Upper surface trailing edge -> leading edge -> Lower surface trailing edge.
        """
        if np.allclose([self.xu[0], self.yu[0]], [self.xl[0], self.yl[0]], atol=1e-10):
            x_selig = np.concatenate([self.xu[::-1], self.xl[1:]])
            y_selig = np.concatenate([self.yu[::-1], self.yl[1:]])
        else:
            x_selig = np.concatenate([self.xu[::-1], self.xl])
            y_selig = np.concatenate([self.yu[::-1], self.yl])
        return x_selig, y_selig

    def save_dat(self, filepath: Union[str, Path], header_name: Optional[str] = None) -> Path:
        """Export coordinates in standard XFOIL / Selig `.dat` format."""
        title = header_name or self.name or self.spec.designation
        return save_dat_coords(filepath, self.coords, header_name=title)

    def save_csv(self, filepath: Union[str, Path]) -> Path:
        """Export full coordinate breakdown to a CSV file."""
        path = Path(filepath)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if len(self.x) == len(self.xu) and len(self.xu) == len(self.xl):
                writer.writerow(["x_c", "yt_c", "yc_c", "dyc_dx", "xu_c", "yu_c", "xl_c", "yl_c"])
                for row in zip(
                    self.x, self.yt, self.yc, self.dyc_dx, self.xu, self.yu, self.xl, self.yl
                ):
                    writer.writerow([f"{val:.6f}" for val in row])
            else:
                writer.writerow(["x", "y"])
                for xi, yi in self.coords:
                    writer.writerow([f"{xi:.6f}", f"{yi:.6f}"])
        return path

    def plot(
        self,
        save_path: Optional[Union[str, Path]] = None,
        show_camber: bool = True,
        show_thickness: bool = False,
    ) -> Optional[Path]:
        """Plot this airfoil with 1:1 aspect ratio."""
        return plot_airfoil(
            self, save_path=save_path, show_camber=show_camber, show_thickness=show_thickness
        )


# Backward-compatible alias
AirfoilGeometry = Airfoil


# =============================================================================
# 4. UNIVERSAL `.dat` I/O & TRAILING-EDGE THICKNESS FUNCTIONS
# =============================================================================

def load_dat_coords(filepath: Union[str, Path]) -> Tuple[np.ndarray, str]:
    """
    Universal `.dat` reader for any airfoil (not limited to NACA 6-series).
    Supports both Selig continuous format (Upper TE -> LE -> Lower TE) and
    Lednicer two-block format (LE -> Upper TE, LE -> Lower TE).

    Returns
    -------
    coords : np.ndarray of shape (N, 2)
        2D array of `[x, y]` coordinates in continuous Selig order.
    title : str
        Header title string from the `.dat` file.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Airfoil .dat file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    title = path.stem
    blocks: List[List[Tuple[float, float]]] = [[]]

    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            if blocks[-1]:
                blocks.append([])
            continue
        parts = s.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                xv, yv = float(parts[0]), float(parts[1])
                if idx <= 2 and xv > 1.5 and yv > 1.5:
                    continue
                blocks[-1].append((xv, yv))
                continue
            except ValueError:
                pass
        if idx == 0:
            title = s

    blocks = [b for b in blocks if len(b) >= 2]
    if not blocks:
        raise ValueError(f"No valid coordinate pairs found in {path}")

    if len(blocks) == 1:
        coords = np.array(blocks[0], dtype=float)
    else:
        upper_pts = np.array(blocks[0], dtype=float)
        lower_pts = np.array(blocks[1], dtype=float)
        if upper_pts[0, 0] > upper_pts[-1, 0]:
            upper_pts = upper_pts[::-1]
        if lower_pts[0, 0] > lower_pts[-1, 0]:
            lower_pts = lower_pts[::-1]
        if np.allclose(upper_pts[0], lower_pts[0], atol=1e-10):
            coords = np.vstack([upper_pts[::-1], lower_pts[1:]])
        else:
            coords = np.vstack([upper_pts[::-1], lower_pts])

    return coords, title


def save_dat_coords(
    filepath: Union[str, Path],
    coords: np.ndarray,
    header_name: str = "Airfoil",
) -> Path:
    """Universal `.dat` writer for any `(N, 2)` airfoil coordinate array."""
    path = Path(filepath)
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected coords of shape (N, 2), got {arr.shape}.")
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{header_name}\n")
        for xi, yi in arr:
            f.write(f"  {xi:10.6f}  {yi:10.6f}\n")
    return path


def load_dat(filepath: Union[str, Path]) -> Airfoil:
    """Convenience wrapper around `Airfoil.from_dat(filepath)`."""
    return Airfoil.from_dat(filepath)


def _te_blend_profile(
    xi: np.ndarray,
    te_blend_power: float = 1.0,
    blend_mode: Literal["power", "xfoil"] = "power",
    xfoil_doc: float = 0.5,
) -> np.ndarray:
    """Evaluate normalized chordwise trailing-edge blending profile `f(xi)` in `[0, 1]`."""
    xi_clamped = np.clip(np.asarray(xi, dtype=float), 0.0, 1.0)
    if blend_mode == "power":
        if te_blend_power < 1.0:
            raise ValueError(
                f"te_blend_power must be >= 1.0 to preserve leading-edge radius, got {te_blend_power}."
            )
        return xi_clamped ** te_blend_power
    elif blend_mode == "xfoil":
        doc = max(float(xfoil_doc), 1e-4)
        return xi_clamped * np.exp(-(1.0 - xi_clamped) / doc)
    else:
        raise ValueError(f"Unknown blend_mode '{blend_mode}'. Choose 'power' or 'xfoil'.")


def apply_te_thickness(
    x: np.ndarray,
    yt: np.ndarray,
    te_thickness: float,
    te_blend_power: float = 1.0,
    blend_mode: Literal["power", "xfoil"] = "power",
) -> np.ndarray:
    """Modify a 1D half-thickness distribution `yt(x)` to achieve `te_thickness` at `x_TE`."""
    if te_thickness < 0.0:
        raise ValueError(f"Trailing-edge thickness te_thickness must be >= 0.0, got {te_thickness}.")

    x_arr = np.asarray(x, dtype=float)
    yt_arr = np.asarray(yt, dtype=float).copy()
    x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
    chord = max(x_max - x_min, 1e-12)
    xi = (x_arr - x_min) / chord

    current_te = 2.0 * float(yt_arr[-1])
    delta_te = te_thickness - current_te

    if math.isclose(delta_te, 0.0, abs_tol=1e-14):
        return yt_arr

    f_xi = _te_blend_profile(xi, te_blend_power=te_blend_power, blend_mode=blend_mode)
    yt_new = yt_arr + 0.5 * delta_te * f_xi
    yt_new[0] = 0.0
    return np.maximum(yt_new, 0.0)


def _modify_upper_lower_arrays(
    xu: np.ndarray,
    yu: np.ndarray,
    xl: np.ndarray,
    yl: np.ndarray,
    te_thickness: float,
    te_blend_power: float = 1.0,
    blend_mode: Literal["power", "xfoil"] = "power",
    relative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Universal geometry kernel that modifies upper `(xu, yu)` and lower `(xl, yl)` arrays."""
    if te_thickness < 0.0:
        raise ValueError(f"te_thickness must be >= 0.0, got {te_thickness}.")

    xu_arr = np.asarray(xu, dtype=float).copy()
    yu_arr = np.asarray(yu, dtype=float).copy()
    xl_arr = np.asarray(xl, dtype=float).copy()
    yl_arr = np.asarray(yl, dtype=float).copy()

    x_le = float(min(xu_arr[0], xl_arr[0]))
    x_te_u = float(xu_arr[-1])
    x_te_l = float(xl_arr[-1])
    chord = max(max(x_te_u, x_te_l) - x_le, 1e-12)

    target_gap = te_thickness * chord if relative else te_thickness
    current_gap = float(yu_arr[-1] - yl_arr[-1])
    delta_gap = target_gap - current_gap

    xi_u = np.clip((xu_arr - x_le) / max(x_te_u - x_le, 1e-12), 0.0, 1.0)
    xi_l = np.clip((xl_arr - x_le) / max(x_te_l - x_le, 1e-12), 0.0, 1.0)

    f_u = _te_blend_profile(xi_u, te_blend_power=te_blend_power, blend_mode=blend_mode)
    f_l = _te_blend_profile(xi_l, te_blend_power=te_blend_power, blend_mode=blend_mode)

    yu_new = yu_arr + 0.5 * delta_gap * f_u
    yl_new = yl_arr - 0.5 * delta_gap * f_l
    return xu_arr, yu_new, xl_arr, yl_new


def modify_airfoil_te_thickness(
    airfoil: Union[
        Airfoil,
        np.ndarray,
        str,
        Path,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    te_thickness: float,
    te_blend_power: float = 1.0,
    blend_mode: Literal["power", "xfoil"] = "power",
    relative: bool = True,
    out_dat: Optional[Union[str, Path]] = None,
    header_name: Optional[str] = None,
) -> Union[
    Airfoil,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Universal function to modify the trailing-edge thickness of an `Airfoil`,
    a `.dat` file, a 2D NumPy coordinate array `(N, 2)` / `(2, N)`, or a tuple
    of coordinate arrays `(x, y)` / `(xu, yu, xl, yl)`.
    """
    if isinstance(airfoil, Airfoil):
        mod_af = airfoil.with_te_thickness(
            te_thickness=te_thickness,
            te_blend_power=te_blend_power,
            blend_mode=blend_mode,
            relative=relative,
        )
        if out_dat is not None:
            mod_af.save_dat(out_dat, header_name=header_name)
        return mod_af

    if isinstance(airfoil, (str, Path)):
        af = Airfoil.from_dat(airfoil).with_te_thickness(
            te_thickness=te_thickness,
            te_blend_power=te_blend_power,
            blend_mode=blend_mode,
            relative=relative,
        )
        if out_dat is not None:
            af.save_dat(out_dat, header_name=header_name)
        return af.coords

    if isinstance(airfoil, tuple):
        if len(airfoil) == 2:
            af = Airfoil.from_numpy(airfoil).with_te_thickness(
                te_thickness=te_thickness,
                te_blend_power=te_blend_power,
                blend_mode=blend_mode,
                relative=relative,
            )
            if out_dat is not None:
                af.save_dat(out_dat, header_name=header_name)
            return af.to_selig_coords()
        elif len(airfoil) == 4:
            af = Airfoil.from_numpy(airfoil).with_te_thickness(
                te_thickness=te_thickness,
                te_blend_power=te_blend_power,
                blend_mode=blend_mode,
                relative=relative,
            )
            if out_dat is not None:
                af.save_dat(out_dat, header_name=header_name)
            return af.xu, af.yu, af.xl, af.yl
        else:
            raise ValueError(f"Expected tuple of length 2 or 4, got {len(airfoil)}.")

    arr = np.asarray(airfoil, dtype=float)
    transposed = arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2
    af = Airfoil.from_numpy(arr).with_te_thickness(
        te_thickness=te_thickness,
        te_blend_power=te_blend_power,
        blend_mode=blend_mode,
        relative=relative,
    )
    if out_dat is not None:
        af.save_dat(out_dat, header_name=header_name)
    return af.coords.T if transposed else af.coords


# =============================================================================
# 5. ANALYTICAL NACA 6-SERIES MEAN CAMBER LINE & THICKNESS INTERPOLATION
# =============================================================================

def compute_mean_camber_line(
    x: np.ndarray,
    c_li: float = 0.2,
    a: float = 1.0,
    modified_6a: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical NACA 6-series or 6A-series mean camber line `y_c/c`
    and slope `dy_c/dx` at dimensionless chord stations `x` in `[0, 1]`.
    """
    x_arr = np.asarray(x, dtype=float)
    yc = np.zeros_like(x_arr)
    dyc = np.zeros_like(x_arr)

    if math.isclose(c_li, 0.0, abs_tol=1e-12):
        return yc, dyc

    if modified_6a:
        scale_6a = 0.9794815
        xm_tan = 0.87437
        te_slope = -0.245209 * c_li

        mask_fwd = x_arr <= xm_tan
        mask_aft = x_arr > xm_tan

        if np.any(mask_fwd):
            yc_f, dyc_f = compute_mean_camber_line(
                x_arr[mask_fwd], c_li=c_li * scale_6a, a=0.8, modified_6a=False
            )
            yc[mask_fwd] = yc_f
            dyc[mask_fwd] = dyc_f

        if np.any(mask_aft):
            yc[mask_aft] = -te_slope * (1.0 - x_arr[mask_aft])
            dyc[mask_aft] = te_slope

        return yc, dyc

    if not (0.0 <= a <= 1.0):
        raise ValueError(f"Mean-line parameter 'a' must be in [0.0, 1.0], got {a}.")

    if math.isclose(a, 1.0, abs_tol=1e-9):
        for i, xi in enumerate(x_arr):
            if xi <= 1e-12 or xi >= 1.0 - 1e-12:
                yc[i] = 0.0
                dyc[i] = 0.0
            else:
                yc[i] = -(c_li / (4.0 * math.pi)) * (
                    (1.0 - xi) * math.log(1.0 - xi) + xi * math.log(xi)
                )
                dyc[i] = (c_li / (4.0 * math.pi)) * (math.log(1.0 - xi) - math.log(xi))
        return yc, dyc

    if math.isclose(a, 0.0, abs_tol=1e-9):
        g = -0.25
    else:
        g = -1.0 / (1.0 - a) * (a**2 * (0.5 * math.log(a) - 0.25) + 0.25)

    h = 1.0 / (1.0 - a) * (0.5 * (1.0 - a)**2 * math.log(1.0 - a) - 0.25 * (1.0 - a)**2) + g
    prefactor = c_li / (2.0 * math.pi * (a + 1.0))

    for i, xi in enumerate(x_arr):
        if xi <= 1e-12 or xi >= 1.0 - 1e-12:
            yc[i] = 0.0
            dyc[i] = 0.0
            continue

        diff_a = a - xi
        if abs(diff_a) < 1e-14:
            term_ax_log = 0.0
            term_ax2_log = 0.0
        else:
            term_ax_log = diff_a * math.log(abs(diff_a))
            term_ax2_log = 0.5 * (diff_a**2) * math.log(abs(diff_a))

        bracket_y = (
            term_ax2_log
            - 0.5 * ((1.0 - xi)**2) * math.log(1.0 - xi)
            + 0.25 * ((1.0 - xi)**2)
            - 0.25 * (diff_a**2)
        )
        yc[i] = prefactor * (bracket_y / (1.0 - a) - xi * math.log(xi) + g - h * xi)

        bracket_dy = (1.0 - xi) * math.log(1.0 - xi) - term_ax_log
        dyc[i] = prefactor * (bracket_dy / (1.0 - a) - math.log(xi) - 1.0 - h)

    return yc, dyc


def _project_surfaces(
    x: np.ndarray,
    yt: np.ndarray,
    yc: np.ndarray,
    dyc_dx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project half-thickness `yt` perpendicular to mean camber line `yc`."""
    theta = np.arctan(dyc_dx)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    xu = x - yt * sin_t
    yu = yc + yt * cos_t
    xl = x + yt * sin_t
    yl = yc - yt * cos_t

    xu[0], yu[0] = 0.0, 0.0
    xl[0], yl[0] = 0.0, 0.0

    if math.isclose(float(x[-1]), 1.0):
        xu[-1], yu[-1] = 1.0, 0.0
        xl[-1], yl[-1] = 1.0, 0.0

    return theta, xu, yu, xl, yl


def compute_thickness_distribution(
    x: np.ndarray,
    family: str = "64",
    t_c: float = 0.12,
) -> Tuple[np.ndarray, float]:
    """
    Compute the standard NACA 6- or 6A-series half-thickness distribution `y_t/c`
    at chordwise stations `x` in `[0, 1]` and the dimensionless leading-edge radius `r_LE/c`.
    """
    fam_key = family.upper().strip()
    if fam_key not in THICKNESS_DATABASE:
        valid = ", ".join(sorted(THICKNESS_DATABASE.keys()))
        raise ValueError(f"Unknown NACA 6-series family '{family}'. Valid families: {valid}")

    if t_c <= 0.0 or t_c >= 1.0 + 1e-9:
        raise ValueError(f"Thickness-to-chord ratio t_c must be in (0, 1.0], got {t_c}.")

    fam_tables = THICKNESS_DATABASE[fam_key]
    avail_t_pct = sorted(fam_tables.keys())
    target_t_pct = t_c * 100.0

    norm_ords = np.array([
        np.array(fam_tables[tp]["y"], dtype=float) / float(tp)
        for tp in avail_t_pct
    ])
    norm_le = np.array([
        float(fam_tables[tp]["le_radius"]) / (float(tp) ** 2)
        for tp in avail_t_pct
    ])

    if len(avail_t_pct) >= 3:
        t_grid = np.array(avail_t_pct, dtype=float)
        t_clamped = float(np.clip(target_t_pct, t_grid[0], t_grid[-1]))
        cs_shape = CubicSpline(t_grid, norm_ords, axis=0, bc_type="natural")
        cs_le = CubicSpline(t_grid, norm_le, bc_type="natural")
        y_std_pct = cs_shape(t_clamped) * target_t_pct
        le_rad_pct = float(cs_le(t_clamped)) * (target_t_pct ** 2)
    else:
        t_grid = np.array(avail_t_pct, dtype=float)
        w = float(np.clip((target_t_pct - t_grid[0]) / (t_grid[-1] - t_grid[0]), 0.0, 1.0))
        shape_interp = (1.0 - w) * norm_ords[0] + w * norm_ords[-1]
        le_interp = (1.0 - w) * norm_le[0] + w * norm_le[-1]
        y_std_pct = shape_interp * target_t_pct
        le_rad_pct = float(le_interp) * (target_t_pct ** 2)

    x_std = np.array(STANDARD_X_PERCENT, dtype=float) / 100.0
    y_std = y_std_pct / 100.0
    le_radius = le_rad_pct / 100.0

    s_std = np.sqrt(x_std)
    dy_ds0 = math.sqrt(max(0.0, 2.0 * le_radius))
    spline_s = CubicSpline(
        s_std,
        y_std,
        bc_type=((1, dy_ds0), "natural"),
    )

    x_arr = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    yt = spline_s(np.sqrt(x_arr))
    yt[0] = 0.0
    if math.isclose(x_arr[-1], 1.0) and abs(y_std[-1]) < 1e-9:
        yt[-1] = 0.0
    yt = np.maximum(yt, 0.0)

    return yt, le_radius


# =============================================================================
# 6. NACA 6-SERIES GENERATOR (`generate_naca6` -> returns `Airfoil`)
# =============================================================================

def generate_naca6(
    designation: Optional[str] = None,
    *,
    family: str = "64",
    c_li: float = 0.2,
    t_c: float = 0.12,
    a: float = 1.0,
    modified_camber: Optional[bool] = None,
    n_points: int = 201,
    spacing: Literal["cosine", "linear", "standard"] = "cosine",
) -> Airfoil:
    """
    Generate a standard NACA 6-series or 6A-series airfoil and return an `Airfoil` object.
    To modify trailing-edge thickness, call `.with_te_thickness(...)` on the returned `Airfoil`.

    Parameters
    ----------
    designation : str, optional
        Compact NACA 6-series string without 'NACA' or spaces, e.g. `"64(1)-212"`,
        `"64-(1.5)12"`, `"64(1)-212,a=0.6"`, `"65(3)-418,a=0.5"`, or `"64A210"`.
    family : str, default "64"
        NACA 6-series family (`"63"`, `"63A"`, `"64"`, `"64A"`, `"65"`, `"65A"`, `"66"`, `"67"`).
    c_li : float, default 0.2
        Design lift coefficient (e.g. `0.2` for `-212`, `0.15` for `-(1.5)12`).
    t_c : float, default 0.12
        Maximum thickness-to-chord ratio (e.g. `0.12` for 12% thickness).
    a : float, default 1.0
        Chordwise fraction of uniform loading (`0.0 <= a <= 1.0`).
    modified_camber : bool, optional
        If True, uses the NACA 6A-series modified `a=0.8` camber line.
    n_points : int, default 201
        Number of chordwise stations from `x/c = 0` to `1` on each surface.
    spacing : {"cosine", "linear", "standard"}, default "cosine"
        Grid spacing rule.

    Returns
    -------
    Airfoil
        Generated `Airfoil` instance.
    """
    if designation is not None:
        spec = parse_naca6_designation(designation)
    else:
        fam_clean = family.upper().strip()
        is_6a = fam_clean.endswith("A") if modified_camber is None else modified_camber
        a_eff = 0.8 if (is_6a and math.isclose(a, 1.0)) else a
        cli_fmt = _format_cli_tenths(c_li)
        tc_fmt = _format_tc_percent(t_c)
        desig_str = (
            f"{fam_clean}{cli_fmt}{tc_fmt}"
            if fam_clean.endswith("A")
            else f"{fam_clean}-{cli_fmt}{tc_fmt}" + (f",a={a_eff:g}" if not math.isclose(a_eff, 1.0) and not math.isclose(c_li, 0.0) else "")
        )
        spec = NACA6Spec(
            family=fam_clean,
            c_li=c_li,
            t_c=t_c,
            a=a_eff,
            modified_camber=is_6a,
            te_thickness=0.0,
            designation=desig_str,
        )

    if spacing == "cosine":
        beta = np.linspace(0.0, math.pi, n_points)
        x = 0.5 * (1.0 - np.cos(beta))
    elif spacing == "linear":
        x = np.linspace(0.0, 1.0, n_points)
    elif spacing == "standard":
        x = np.array(STANDARD_X_PERCENT, dtype=float) / 100.0
    else:
        raise ValueError(f"Unknown spacing mode '{spacing}'. Choose 'cosine', 'linear', or 'standard'.")

    yt, le_radius = compute_thickness_distribution(
        x,
        family=spec.family,
        t_c=spec.t_c,
    )
    yc, dyc_dx = compute_mean_camber_line(
        x, c_li=spec.c_li, a=spec.a, modified_6a=spec.modified_camber
    )

    theta, xu, yu, xl, yl = _project_surfaces(x, yt, yc, dyc_dx)

    return Airfoil(
        xu=xu,
        yu=yu,
        xl=xl,
        yl=yl,
        name=spec.designation,
        x=x,
        yt=yt,
        yc=yc,
        dyc_dx=dyc_dx,
        theta=theta,
        le_radius=le_radius,
        spec=spec,
    )


# =============================================================================
# 7. PLOTTING UTILITY & CLI ENTRYPOINT
# =============================================================================

def plot_airfoil(
    airfoils: Union[Airfoil, Sequence[Airfoil]],
    save_path: Optional[Union[str, Path]] = None,
    show_camber: bool = True,
    show_thickness: bool = False,
) -> Optional[Path]:
    """Plot one or more `Airfoil` objects with true 1:1 aspect ratio."""
    import matplotlib.pyplot as plt

    if isinstance(airfoils, Airfoil):
        airfoil_list = [airfoils]
    else:
        airfoil_list = list(airfoils)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    palette = ["#0F4C81", "#D9534F", "#2A9D8F", "#E9C46A", "#6A4C93"]

    for idx, af in enumerate(airfoil_list):
        color = palette[idx % len(palette)]
        x_s, y_s = af.to_selig_coords()
        te_tag = f", t_TE={af.actual_te_thickness*100:.2f}%c" if af.actual_te_thickness > 1e-7 else ""
        le_tag = f"r_LE={af.le_radius*100:.3f}%c" if af.le_radius > 0 else f"t/c={af.spec.t_c*100:.1f}%"
        ax.plot(
            x_s, y_s, color=color, linewidth=2.0,
            label=f"{af.name} ({le_tag}{te_tag})"
        )
        ax.fill(x_s, y_s, color=color, alpha=0.08)

        if show_camber and np.max(np.abs(af.yc)) > 1e-6:
            ax.plot(
                af.x, af.yc, color=color, linestyle="--", linewidth=1.2,
                alpha=0.85, label=f"{af.name} Camber Line"
            )
        if show_thickness:
            ax.plot(
                af.x, af.yt, color=color, linestyle=":", linewidth=1.1,
                alpha=0.75, label=f"{af.name} Half-Thickness y_t/c"
            )

    ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("x / c", fontsize=11, fontweight="bold")
    ax.set_ylabel("y / c", fontsize=11, fontweight="bold")
    title_str = (
        f"Airfoil Profile: {airfoil_list[0].name}"
        if len(airfoil_list) == 1
        else "Airfoil Profile Comparison"
    )
    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        out_path = Path(save_path)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return out_path

    plt.show()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Airfoil tool & NACA 6-series / 6A-series coordinate generator."
    )
    parser.add_argument(
        "designation",
        nargs="?",
        default="64(1)-212,a=0.6",
        help="Compact 6-series designation (e.g. '64(1)-212,a=0.6', '64-(1.5)12') or path to a .dat file.",
    )
    parser.add_argument(
        "-t", "--te-thickness", type=float, default=None,
        help="Trailing-edge thickness t_TE/c as fraction of chord (e.g. 0.002 for 0.2%% chord)."
    )
    parser.add_argument(
        "--te-blend-power", type=float, default=1.0,
        help="Chordwise blending exponent p for trailing-edge thickness (x/c)^p (default: 1.0)."
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize the airfoil coordinates into [0, 1] chord interval."
    )
    parser.add_argument("-n", "--points", type=int, default=201, help="Number of chordwise points per surface (default: 201).")
    parser.add_argument(
        "-s", "--spacing", choices=["cosine", "linear", "standard"], default="cosine",
        help="Chordwise point distribution (default: cosine)."
    )
    parser.add_argument("-o", "--out-dat", type=str, default=None, help="Output Selig/XFOIL .dat file path.")
    parser.add_argument("--out-csv", type=str, default=None, help="Output detailed CSV table path.")
    parser.add_argument("--plot", type=str, default=None, help="Save airfoil plot to PNG/PDF path.")
    args = parser.parse_args()

    if Path(args.designation).is_file() or args.designation.lower().endswith(".dat"):
        af = Airfoil.from_dat(args.designation)
        if args.normalize:
            af = af.normalize()
        if args.te_thickness is not None:
            af = af.with_te_thickness(
                te_thickness=args.te_thickness, te_blend_power=args.te_blend_power
            )
    else:
        af = generate_naca6(
            args.designation,
            n_points=args.points,
            spacing=args.spacing,
        )
        if args.normalize:
            af = af.normalize()
        if args.te_thickness is not None:
            af = af.with_te_thickness(
                te_thickness=args.te_thickness, te_blend_power=args.te_blend_power
            )

    print(f"Airfoil             : {af.name}")
    print(f"  Class Type        : {type(af).__name__}")
    print(f"  Family            : {af.spec.family}")
    print(f"  Chord Length      : {af.chord:.6f}")
    print(f"  Max Thickness t/c : {af.spec.t_c*100:.2f}% chord")
    print(f"  Trailing-Edge Gap : {af.actual_te_thickness*100:.4f}% chord (dy_TE = {af.actual_te_thickness:.6f})")
    if af.le_radius > 0:
        print(f"  Leading-Edge Rad. : {af.le_radius*100:.4f}% chord")
    print(f"  Max Camber y_c/c  : {np.max(af.yc)*100:.4f}% chord")
    print("-" * 68)
    print(f"{'x/c':>8} | {'yt/c':>9} | {'yc/c':>9} | {'xu/c':>9} | {'yu/c':>9} | {'xl/c':>9} | {'yl/c':>9}")
    print("-" * 68)

    sample_indices = np.linspace(0, len(af.x) - 1, min(11, len(af.x)), dtype=int)
    for idx in sample_indices:
        print(
            f"{af.x[idx]:8.4f} | {af.yt[idx]:9.5f} | {af.yc[idx]:9.5f} | "
            f"{af.xu[idx]:9.5f} | {af.yu[idx]:9.5f} | {af.xl[idx]:9.5f} | {af.yl[idx]:9.5f}"
        )

    if args.out_dat:
        p = af.save_dat(args.out_dat)
        print(f"\nSaved XFOIL/Selig .dat file -> {p}")
    if args.out_csv:
        p = af.save_csv(args.out_csv)
        print(f"Saved CSV coordinate table  -> {p}")
    if args.plot:
        p = af.plot(save_path=args.plot)
        print(f"Saved airfoil plot          -> {p}")


if __name__ == "__main__":
    # main()
    af=generate_naca6("64-(1.5)12").with_te_thickness(0.002)
    af.save_dat("./generate_naca6.dat")