"""XMA2 (Xbox 360) encoding and analysis for the FSB5 sound banks.

XMA2 is WMA Pro with 512 samples per frame, in 2048-byte packets. What follows is derived from the
sources of the FFmpeg decoder (libavcodec/wmaprodec.c, tables from wmaprodata.h) and from the analysis
of the XMA files of the 360 game.

License: the coding tables and the frame reading derive from FFmpeg (libavcodec/wmaprodata.h and
wmaprodec.c), distributed under the GNU LGPL 2.1 or later; this file is distributed under the same
terms.

Packet: 32-bit big endian header (6 bits frames starting in the packet, 15 bits bit position of the
first of these frames after the header, 3 bits metadata = 1, 8 bits skip = 0), then 2044 bytes in
which the frames are written one after the other, also across packets. A block is at most 16 packets
(32 KB): frames do not cross it and the tail is filled with 1 bits.

Frame (FFmpeg: decode_frame): 15 bits total length, tile header, (stereo) 1 bit postproc, 8 bits DRC,
trim, four 128-sample subframes, 1 bit set to 1 and 1 bit "another frame in the same packet".
"""
import os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# FFmpeg tables (libavcodec/wmaprodata.h): critical bands, Huffman codes of the scale factors,
# run-level and vector coefficients (symbol, length pairs or parallel lists).
CRITICAL_FREQ = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
    3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20675, 28575, 41375, 63875]
SCALE_TABLE = [(58, 5), (64, 6), (66, 7), (65, 7), (62, 5), (63, 6), (68, 9), (69, 10), (54, 15), (19, 19),
    (20, 19), (21, 19), (22, 19), (23, 19), (24, 19), (25, 19), (26, 19), (27, 19), (28, 19), (29, 19), (30,
    19), (31, 19), (32, 19), (33, 19), (34, 19), (17, 19), (36, 19), (37, 19), (38, 19), (39, 19), (40, 19),
    (41, 19), (42, 19), (43, 19), (44, 19), (45, 19), (46, 19), (47, 19), (48, 19), (49, 19), (50, 19), (51,
    19), (52, 19), (15, 19), (16, 19), (14, 19), (13, 19), (12, 19), (11, 19), (10, 19), (0, 19), (9, 19),
    (8, 19), (7, 19), (6, 19), (5, 19), (4, 19), (55, 13), (70, 13), (3, 19), (2, 19), (1, 19), (35, 19),
    (71, 19), (72, 19), (73, 19), (74, 19), (75, 19), (76, 19), (77, 19), (78, 19), (79, 19), (80, 19), (81,
    19), (82, 19), (83, 19), (84, 19), (85, 19), (86, 19), (87, 19), (88, 19), (89, 19), (90, 19), (91, 19),
    (92, 19), (93, 19), (94, 19), (95, 19), (96, 19), (97, 19), (98, 19), (99, 19), (100, 19), (101, 19),
    (102, 19), (103, 19), (104, 19), (105, 19), (106, 19), (107, 19), (108, 19), (109, 19), (110, 19), (111,
    19), (112, 19), (113, 19), (114, 19), (115, 19), (116, 19), (117, 19), (118, 19), (119, 19), (120, 19),
    (18, 18), (53, 16), (56, 11), (57, 8), (67, 7), (61, 3), (59, 2), (60, 1)]
SCALE_RL_TABLE = [(103, 7), (80, 11), (60, 11), (18, 10), (56, 10), (21, 12), (90, 12), (58, 11), (27, 11),
    (69, 12), (84, 15), (48, 15), (86, 14), (47, 13), (19, 10), (32, 9), (78, 6), (5, 5), (28, 4), (53, 5),
    (9, 7), (31, 8), (38, 8), (10, 7), (88, 11), (25, 12), (105, 12), (118, 11), (23, 12), (82, 14), (98,
    16), (110, 16), (108, 15), (93, 13), (68, 10), (72, 12), (97, 12), (81, 12), (42, 12), (64, 8), (4, 4),
    (1, 2), (7, 6), (14, 7), (0, 9), (55, 9), (61, 9), (117, 10), (24, 12), (44, 12), (67, 12), (70, 16),
    (99, 18), (96, 21), (95, 21), (2, 21), (77, 21), (52, 21), (111, 21), (102, 20), (101, 17), (46, 15),
    (73, 15), (109, 15), (51, 14), (92, 14), (30, 7), (11, 7), (66, 7), (15, 8), (16, 8), (116, 9), (65, 9),
    (57, 10), (59, 10), (115, 9), (12, 7), (35, 9), (17, 9), (41, 9), (20, 11), (91, 11), (26, 12), (75,
    15), (45, 15), (107, 14), (83, 14), (100, 15), (89, 15), (43, 11), (62, 9), (37, 9), (104, 8), (6, 5),
    (39, 8), (40, 9), (34, 9), (79, 7), (8, 6), (63, 6), (87, 12), (94, 14), (49, 14), (50, 13), (22, 11),
    (119, 10), (33, 9), (36, 9), (113, 11), (106, 12), (112, 13), (71, 15), (85, 15), (74, 14), (76, 10),
    (114, 7), (29, 5), (54, 6), (13, 6), (3, 2)]
SCALE_RL_RUN = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    0, 1, 0, 1, 0, 1]
SCALE_RL_LEVEL = [0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 8, 8, 9, 9]
COEF0_LENS = [2, 9, 14, 14, 13, 12, 13, 14, 15, 15, 12, 10, 10, 10, 13, 14, 15, 15, 12, 11, 13, 14, 14, 13,
    15, 15, 14, 12, 12, 8, 10, 10, 15, 15, 14, 13, 14, 14, 13, 15, 20, 20, 19, 21, 21, 20, 19, 17, 17, 18,
    18, 15, 15, 13, 12, 14, 15, 15, 14, 15, 15, 12, 11, 6, 7, 8, 9, 13, 13, 13, 14, 14, 11, 10, 7, 8, 14,
    14, 14, 14, 12, 13, 13, 12, 12, 12, 11, 9, 13, 14, 14, 12, 11, 11, 11, 9, 8, 7, 14, 15, 15, 14, 14, 12,
    13, 15, 16, 17, 17, 14, 12, 12, 12, 15, 15, 14, 14, 14, 13, 13, 9, 9, 11, 11, 10, 7, 6, 13, 15, 15, 14,
    14, 14, 13, 14, 15, 15, 13, 14, 14, 14, 14, 10, 9, 10, 10, 11, 11, 10, 8, 9, 13, 14, 14, 12, 11, 14, 15,
    15, 13, 12, 14, 14, 14, 14, 13, 14, 14, 3, 5, 8, 10, 10, 15, 15, 14, 14, 16, 16, 15, 12, 11, 11, 11, 7,
    8, 8, 9, 12, 13, 13, 12, 14, 15, 15, 13, 10, 11, 11, 13, 14, 14, 13, 14, 14, 11, 10, 13, 15, 15, 14, 12,
    11, 4, 6, 6, 8, 12, 12, 12, 13, 13, 12, 13, 13, 14, 14, 13, 13, 13, 9, 7, 9, 11, 14, 14, 13, 14, 14, 13,
    10, 8, 7, 5, 9, 12, 13, 14, 15, 15, 12, 12, 10, 14, 14, 13, 12, 13, 14, 14, 12, 13, 13, 12, 12, 12, 9,
    7, 6, 3, 4, 4]
COEF0_SYMS = [2, 25, 111, 94, 69, 58, 87, 93, 136, 135, 59, 37, 34, 36, 82, 182, 120, 138, 195, 45, 168,
    216, 178, 86, 140, 219, 186, 162, 239, 18, 156, 35, 127, 236, 109, 85, 180, 253, 88, 147, 268, 264, 256,
    266, 270, 262, 260, 248, 246, 252, 258, 137, 189, 230, 64, 179, 146, 208, 101, 118, 238, 163, 46, 9,
    153, 0, 26, 247, 169, 76, 202, 131, 194, 38, 13, 19, 132, 106, 191, 97, 65, 198, 77, 62, 66, 164, 48,
    27, 81, 183, 102, 60, 47, 49, 159, 227, 20, 14, 112, 263, 144, 217, 104, 63, 79, 209, 269, 250, 254,
    203, 241, 196, 61, 220, 148, 124, 185, 100, 80, 78, 193, 28, 50, 235, 41, 1, 10, 171, 226, 150, 103,
    114, 115, 170, 105, 211, 149, 249, 108, 188, 107, 255, 231, 155, 42, 40, 55, 160, 39, 21, 29, 215, 234,
    184, 228, 51, 116, 142, 145, 172, 165, 181, 130, 113, 117, 89, 128, 204, 3, 7, 154, 157, 43, 141, 265,
    133, 225, 271, 244, 221, 74, 54, 56, 52, 15, 222, 22, 30, 83, 199, 173, 73, 123, 210, 143, 175, 44, 53,
    237, 174, 139, 134, 110, 218, 129, 161, 213, 177, 267, 151, 125, 67, 223, 5, 11, 192, 23, 214, 243, 166,
    200, 176, 68, 224, 187, 257, 261, 232, 96, 251, 31, 16, 32, 57, 207, 121, 91, 126, 119, 99, 158, 24,
    212, 8, 33, 70, 92, 205, 240, 242, 75, 197, 233, 259, 190, 98, 71, 201, 122, 206, 72, 90, 95, 84, 167,
    245, 229, 17, 12, 4, 152, 6]
COEF1_TABLE = [(2, 2), (3, 3), (102, 3), (4, 4), (148, 6), (134, 9), (171, 10), (18, 10), (11, 8), (159, 8),
    (14, 9), (156, 14), (235, 15), (61, 15), (38, 13), (153, 13), (48, 14), (49, 14), (23, 11), (203, 13),
    (208, 19), (204, 19), (129, 18), (94, 17), (87, 16), (62, 15), (174, 15), (147, 15), (29, 12), (191,
    12), (64, 15), (65, 15), (146, 14), (164, 13), (142, 5), (132, 4), (103, 5), (154, 7), (165, 9), (181,
    11), (109, 12), (30, 12), (86, 16), (92, 16), (239, 15), (138, 14), (39, 13), (50, 14), (115, 15), (238,
    21), (228, 21), (236, 21), (222, 21), (216, 20), (226, 20), (196, 18), (192, 17), (120, 16), (221, 14),
    (51, 14), (24, 11), (143, 8), (7, 6), (9, 7), (152, 10), (136, 12), (160, 12), (241, 15), (66, 15),
    (168, 14), (219, 14), (113, 14), (193, 12), (19, 10), (173, 10), (105, 8), (149, 9), (15, 9), (205, 13),
    (207, 13), (125, 17), (190, 17), (182, 16), (68, 15), (70, 15), (67, 15), (137, 13), (31, 12), (223,
    14), (116, 15), (210, 19), (220, 19), (198, 18), (126, 17), (88, 16), (41, 13), (25, 11), (40, 13), (73,
    15), (243, 15), (53, 14), (195, 12), (183, 11), (225, 14), (52, 14), (71, 15), (121, 16), (89, 16),
    (170, 14), (55, 14), (69, 15), (83, 15), (209, 13), (108, 11), (32, 12), (54, 14), (122, 16), (184, 16),
    (176, 15), (42, 13), (12, 8), (161, 8), (6, 5), (167, 9), (106, 9), (20, 10), (145, 12), (111, 13), (43,
    13), (26, 11), (175, 10), (107, 10), (34, 12), (33, 12), (197, 12), (74, 15), (128, 17), (232, 20),
    (212, 20), (224, 19), (202, 18), (90, 16), (57, 14), (227, 14), (97, 16), (93, 16), (140, 15), (185,
    11), (27, 11), (16, 9), (158, 11), (211, 13), (56, 14), (117, 15), (72, 15), (166, 13), (91, 16), (95,
    16), (80, 15), (101, 16), (194, 17), (127, 17), (82, 15), (21, 10), (144, 10), (177, 10), (151, 6), (10,
    7), (157, 7), (8, 6), (5, 4), (13, 8), (0, 9), (213, 13), (46, 13), (199, 12), (35, 12), (162, 12),
    (135, 10), (169, 9), (45, 13), (59, 14), (114, 14), (44, 13), (188, 16), (186, 16), (75, 15), (79, 15),
    (118, 15), (187, 11), (112, 13), (139, 14), (178, 15), (81, 15), (110, 12), (28, 11), (163, 8), (133,
    6), (104, 6), (17, 9), (22, 10), (229, 14), (172, 14), (217, 13), (201, 12), (36, 12), (218, 20), (242,
    22), (240, 22), (234, 21), (230, 19), (206, 18), (200, 18), (214, 18), (130, 17), (131, 17), (141, 15),
    (84, 15), (76, 15), (215, 13), (58, 14), (231, 14), (233, 14), (180, 15), (77, 15), (37, 12), (189, 11),
    (179, 10), (155, 10), (47, 13), (96, 16), (99, 16), (119, 15), (63, 14), (237, 14), (78, 15), (85, 15),
    (60, 14), (98, 16), (100, 16), (124, 16), (123, 16), (150, 11), (1, 7)]
COEF0_RUN = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0]
COEF0_LEVEL = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
    25, 25, 26, 26, 27, 27, 28]
COEF1_RUN = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0, 1,
    2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 0, 0]
COEF1_LEVEL = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
    26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39,
    39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 51, 52]
VEC4_LENS = [1, 6, 8, 10, 10, 10, 10, 8, 8, 10, 10, 9, 8, 8, 9, 12, 12, 11, 12, 12, 11, 9, 9, 8, 8, 9, 9, 8,
    8, 9, 9, 12, 12, 12, 14, 14, 13, 11, 11, 9, 8, 9, 9, 11, 11, 10, 9, 8, 6, 6, 6, 6, 6, 6, 11, 11, 10, 11,
    11, 10, 10, 11, 11, 9, 7, 6, 7, 7, 6, 6, 6, 5, 7, 11, 11, 10, 9, 8, 6, 9, 9, 10, 10, 9, 8, 8, 6, 6, 6,
    8, 8, 9, 12, 12, 11, 10, 8, 8, 8, 10, 10, 10, 10, 9, 9, 8, 10, 11, 11, 9, 8, 8, 8, 9, 9, 10, 11, 12, 12,
    9, 9, 9, 8, 8, 7, 7, 7]
VEC4_SYMS = [0, 4370, 275, 8195, 4146, 12545, 8225, 290, 4625, 515, 20, 8706, 8210, 4355, 4131, 16385, 5121,
    8961, 321, 1041, 51, 4641, 546, 4610, 530, 513, 8451, 4385, 4130, 33, 8211, 5, 66, 4161, 1281, 81, 6,
    801, 8196, 8481, 8449, 4611, 531, 561, 769, 12290, 8226, 19, 4097, 2, 4369, 274, 4354, 4114, 12291,
    16641, 12305, 49, 12321, 260, 4100, 516, 21, 12546, 8466, 4353, 4371, 4626, 257, 18, 17, 1, 4386, 8241,
    771, 4865, 8705, 8194, 4098, 12561, 276, 50, 785, 4116, 8209, 4099, 273, 4113, 258, 259, 4609, 35, 1026,
    1025, 16401, 305, 34, 529, 289, 770, 12289, 4, 4145, 4356, 12306, 8193, 12801, 261, 16386, 4881, 3, 514,
    4129, 545, 306, 36, 4101, 65, 20481, 786, 4401, 4866, 8721, 291, 8450, 8465, 4115]
VEC2_TABLE = [(19, 5), (165, 10), (211, 11), (46, 11), (75, 10), (177, 11), (12, 11), (86, 8), (83, 7), (38,
    7), (133, 9), (178, 10), (28, 10), (104, 9), (73, 9), (35, 5), (52, 6), (113, 9), (8, 9), (101, 8), (69,
    7), (0, 3), (71, 8), (119, 9), (91, 10), (179, 10), (114, 8), (166, 10), (10, 10), (44, 10), (145, 10),
    (66, 6), (21, 6), (24, 8), (146, 9), (26, 9), (65, 7), (5, 7), (226, 11), (225, 12), (15, 12), (180,
    10), (147, 9), (115, 8), (40, 8), (89, 9), (134, 9), (84, 7), (54, 7), (42, 9), (60, 10), (31, 11),
    (193, 11), (181, 10), (76, 10), (148, 9), (37, 6), (67, 6), (33, 6), (3, 6), (17, 6), (2, 6), (102, 8),
    (87, 8), (116, 8), (56, 8), (50, 5), (20, 5), (120, 9), (58, 9), (29, 10), (194, 10), (135, 9), (97, 8),
    (7, 8), (105, 9), (13, 11), (241, 12), (16, 12), (45, 10), (149, 9), (74, 9), (98, 7), (23, 7), (85, 7),
    (70, 7), (195, 10), (161, 10), (129, 9), (72, 8), (51, 5), (36, 5), (117, 8), (61, 10), (11, 10), (162,
    9), (1, 7), (4, 6), (49, 6), (68, 6), (9, 9), (27, 9), (130, 8), (39, 7), (53, 6), (99, 7), (25, 8),
    (150, 9), (90, 9), (103, 8), (163, 9), (196, 10), (210, 10), (136, 9), (121, 9), (41, 8), (131, 8), (43,
    9), (164, 9), (118, 8), (88, 8), (81, 7), (6, 7), (55, 7), (59, 9), (30, 10), (209, 11), (14, 11), (151,
    9), (106, 9), (82, 6), (22, 6), (100, 7), (132, 8), (57, 8), (18, 4), (34, 4)]
VEC1_TABLE = [(7, 5), (32, 8), (59, 10), (60, 10), (83, 11), (82, 11), (62, 10), (33, 8), (45, 9), (61, 10),
    (84, 11), (85, 11), (1, 6), (13, 5), (19, 6), (25, 7), (34, 8), (46, 9), (47, 9), (14, 5), (6, 5), (64,
    10), (87, 11), (86, 11), (63, 10), (88, 11), (90, 11), (35, 8), (26, 7), (0, 7), (48, 9), (65, 10), (66,
    10), (36, 8), (15, 5), (20, 6), (91, 11), (89, 11), (67, 10), (49, 9), (50, 9), (69, 10), (92, 11), (93,
    11), (27, 7), (5, 5), (37, 8), (68, 10), (71, 10), (51, 9), (52, 9), (70, 10), (94, 11), (96, 11), (38,
    8), (21, 6), (16, 5), (4, 5), (28, 7), (53, 9), (95, 11), (97, 11), (73, 10), (39, 8), (29, 7), (72,
    10), (98, 11), (99, 11), (54, 9), (40, 8), (22, 6), (30, 7), (55, 9), (74, 10), (76, 10), (56, 9), (75,
    10), (77, 10), (17, 5), (3, 5), (23, 6), (41, 8), (57, 9), (78, 10), (79, 10), (31, 7), (10, 4), (9, 4),
    (100, 5), (2, 5), (11, 4), (8, 4), (18, 5), (42, 8), (58, 9), (80, 10), (81, 10), (43, 8), (44, 8), (24,
    6), (12, 4)]

SPF = 512          # samples per frame
SUB = 128          # samples per subframe (the 360 game always uses 4 x 128)
PKT = 2048
PB = (PKT - 4) * 8  # usable bits per packet
BLOCK_PACKETS = 16


def _vlc(lens, syms, offset=0):
    """Canonical codes as in ff_vlc_init_from_lengths: in table order, left aligned."""
    code = 0; enc = {}; dec = {}
    for l, s in zip(lens, syms):
        c = code >> (32 - l); sym = s + offset
        if sym not in enc or enc[sym][1] > l:
            enc[sym] = (c, l)
        dec[(l, c)] = sym
        code += 1 << (32 - l)
    assert code == 1 << 32
    return enc, dec, max(lens)


SF_VLC = _vlc([l for s, l in SCALE_TABLE], [s for s, l in SCALE_TABLE], -60)
SFRL_VLC = _vlc([l for s, l in SCALE_RL_TABLE], [s for s, l in SCALE_RL_TABLE])
COEF_VLC = (_vlc(COEF0_LENS, COEF0_SYMS), _vlc([l for s, l in COEF1_TABLE], [s for s, l in COEF1_TABLE]))
COEF_RUN = (COEF0_RUN, COEF1_RUN)
COEF_LEVEL = (COEF0_LEVEL, COEF1_LEVEL)
V4_VLC = _vlc(VEC4_LENS, VEC4_SYMS, -1)
V2_VLC = _vlc([l for s, l in VEC2_TABLE], [s for s, l in VEC2_TABLE], -1)
V1_VLC = _vlc([l for s, l in VEC1_TABLE], [s for s, l in VEC1_TABLE])

# (run, level) -> code index, for the two run-level coefficient tables
RL_INDEX = []
for t in range(2):
    m = {}
    for code in range(2, len(COEF_RUN[t])):
        key = (COEF_RUN[t][code], COEF_LEVEL[t][code])
        if key not in m or COEF_VLC[t][0][code][1] < COEF_VLC[t][0][m[key]][1]:
            m[key] = code
    RL_INDEX.append(m)
SFRL_INDEX = {}
for idx in range(2, len(SCALE_RL_RUN)):
    key = (SCALE_RL_RUN[idx], SCALE_RL_LEVEL[idx])
    if key not in SFRL_INDEX or SFRL_VLC[0][idx][1] < SFRL_VLC[0][SFRL_INDEX[key]][1]:
        SFRL_INDEX[key] = idx
V4_SYM = {}
for sym in V4_VLC[0]:
    if sym >= 0:
        V4_SYM[(sym >> 12, (sym >> 8) & 15, (sym >> 4) & 15, sym & 15)] = sym
V2_SYM = {(s >> 4, s & 15): s for s in V2_VLC[0] if s >= 0}


def band_offsets(rate=32000):
    """Scale factor band limits for 512, 256, 128 subframes (FFmpeg decode_init)."""
    r = 48000 if rate > 44100 else 44100 if rate > 32000 else 32000 if rate > 24000 else 24000
    res = []
    for i in range(3):
        sl = SPF >> i; o = [0]
        for x in range(28):
            off = ((sl * 2 * CRITICAL_FREQ[x]) // r + 2) & ~3
            if off > o[-1]: o.append(off)
            if off >= sl: break
        o[-1] = sl
        res.append(o)
    return res


def sf_resample_table(offs):
    n = len(offs); m = [[None] * n for _ in range(n)]
    for i in range(n):
        for x in range(n):
            row = []
            for b in range(len(offs[i]) - 1):
                o = ((offs[i][b] + offs[i][b + 1] - 1) << i) >> 1
                v = 0
                while (offs[x][v + 1] << x) < o: v += 1
                row.append(v)
            m[i][x] = row
    return m


# ---------------------------------------------------------------------------------------------
# bit writing

class BitWriter:
    __slots__ = ('acc', 'n')

    def __init__(self):
        self.acc = 0; self.n = 0

    def put(self, v, k):
        if k:
            self.acc = (self.acc << k) | (v & ((1 << k) - 1)); self.n += k

    def code(self, vlc, sym):
        c, l = vlc[0][sym]; self.put(c, l)

    def large(self, v):
        if v < 1 << 8: self.put(0, 1); self.put(v, 8)
        elif v < 1 << 16: self.put(2, 2); self.put(v, 16)
        elif v < 1 << 24: self.put(6, 3); self.put(v, 24)
        else: self.put(7, 3); self.put(v, 31)

    def bits(self):
        return format(self.acc, '0%db' % self.n) if self.n else ''


def write_coefs(bw, q, tab, tnv, nvc):
    """Integer coefficients of a channel (FFmpeg decode_coeffs). q: list of ints as long as the subframe."""
    sl = len(q); bw.put(tab, 1)
    cur = 0; rl = 0; nz = 0; thr = sl >> 8
    while (tnv or not rl) and cur + 3 < nvc:
        g = q[cur:cur + 4]; a = tuple(abs(v) for v in g)
        sym = V4_SYM.get(a)
        if sym is not None:
            bw.code(V4_VLC, sym)
        else:
            bw.code(V4_VLC, -1)
            for i in (0, 2):
                s2 = V2_SYM.get(a[i:i + 2])
                if s2 is not None:
                    bw.code(V2_VLC, s2)
                else:
                    bw.code(V2_VLC, -1)
                    for v in a[i:i + 2]:
                        if v < 100: bw.code(V1_VLC, v)
                        else: bw.code(V1_VLC, 100); bw.large(v - 100)
        for v in g:
            if v:
                bw.put(1 if v > 0 else 0, 1); nz = 0
            else:
                nz += 1
                if nz > thr: rl = 1
        cur += 4
    if cur < sl:
        vlc = COEF_VLC[tab]; idx = RL_INDEX[tab]; esc_len = (sl - 1).bit_length()
        pos = cur
        for k in range(cur, sl):
            v = q[k]
            if not v: continue
            run = k - pos; a = abs(v)
            code = idx.get((run, a))
            if code is not None:
                bw.code(vlc, code)
            else:
                bw.code(vlc, 0); bw.large(a)
                if run == 0: bw.put(0, 1)
                elif run <= 4: bw.put(2, 2); bw.put(run - 1, 2)
                else: bw.put(6, 3); bw.put(run - 4, esc_len)
            bw.put(1 if v > 0 else 0, 1)
            pos = k + 1
        if pos < sl:
            bw.code(vlc, 1)


def write_q(bw, q):
    d = q - 90
    if -32 < d < 31:
        bw.put(d & 63, 6)
    elif d >= 31:
        bw.put(31, 6); r = d - 31
        while r >= 31: bw.put(31, 5); r -= 31
        bw.put(r, 5)
    else:
        bw.put(32, 6); r = -32 - d
        while r >= 31: bw.put(31, 5); r -= 31
        bw.put(r, 5)


def write_frame(fr, nch, offs):
    """Serializes a frame described as a dict (see parse_frame). Returns the bit string
    with the final "another frame" bit set to 1 (fixed by the packing)."""
    bw = BitWriter()
    tiles = fr.get('tiles', [[SUB] * 4] * nch)
    bw.put(1, 1)  # same split for every channel
    t = tiles[0]; pos = 0
    for sl in t:
        if pos == SPF - SUB: break
        sh = {512: 0, 256: 1, 128: 2}[sl]
        if sh: bw.put(1, 1); bw.put(sh - 1, 1)
        else: bw.put(0, 1)
        pos += sl
    if nch > 1:
        bw.put(fr.get('postproc', 0), 1)
    bw.put(fr.get('drc', 0), 8)
    ts, te = fr.get('trim', (None, None))
    if ts is None and te is None:
        bw.put(0, 1)
    else:
        bw.put(1, 1)
        if ts is None: bw.put(0, 1)
        else: bw.put(1, 1); bw.put(ts, 10)
        if te is None: bw.put(0, 1)
        else: bw.put(1, 1); bw.put(te, 10)
    reuse = [0] * nch
    for si, sub in enumerate(fr['subs']):
        chans = sub['chans']; sl = sub['len']
        ti = {512: 0, 256: 1, 128: 2}[sl]; nb = len(offs[ti]) - 1
        bw.put(0, 1)   # no extended header
        bw.put(0, 1)   # reserved bit
        if nch > 1:
            bw.put(0, 1)
            if len(chans) == 2:
                if sub['tr'] == 'none':
                    bw.put(2, 2)
                else:
                    bw.put(0, 1)
                    if sub['tb'] == 'all': bw.put(1, 1)
                    else:
                        bw.put(0, 1)
                        for b in sub['tb']: bw.put(b, 1)
        for c in chans: bw.put(sub['tc'][c], 1)
        if any(sub['tc'][c] for c in chans):
            bw.put(sub['tnv'], 1)
            if sub['tnv']:
                nbits = ((sl + 3) // 4).bit_length()
                for c in chans: bw.put(sub['nvc'][c] >> 2, nbits)
            write_q(bw, sub['q'])
            if len(chans) > 1:
                ml, mods = sub.get('mod', (0, {}))
                bw.put(ml, 3)
                for c in chans:
                    m = mods.get(c)
                    if m is None: bw.put(0, 1)
                    else:
                        bw.put(1, 1)
                        if ml: bw.put(m - 1, ml)
            for c in chans:
                mode = sub['sf'][c]
                if si > 0:
                    if mode[0] == 'keep': bw.put(0, 1); continue
                    bw.put(1, 1)
                if mode[0] == 'dpcm':
                    assert not reuse[c]
                    step, vals = mode[1], mode[2]
                    bw.put(step - 1, 2); prev = 45 // step
                    for v in vals:
                        bw.code(SF_VLC, v - prev); prev = v
                else:
                    assert mode[0] == 'rl' and reuse[c]
                    i = 0
                    for b, d in mode[1]:
                        skip = b - i; a = abs(d)
                        idx = SFRL_INDEX.get((skip, a))
                        if idx is not None:
                            bw.code(SFRL_VLC, idx); bw.put(1 if d > 0 else 0, 1)
                        else:
                            bw.code(SFRL_VLC, 0)
                            bw.put((a << 6) | (skip << 1) | (1 if d > 0 else 0), 14)
                        i = b + 1
                    if i < nb: bw.code(SFRL_VLC, 1)
                reuse[c] = 1
        for c in chans:
            if sub['tc'][c]:
                tab, q = sub['coefs'][c]
                write_coefs(bw, q, tab, sub['tnv'], sub['nvc'][c] if sub['tnv'] else sl)
    total = bw.n + 17
    assert total < PB
    return format(total, '015b') + bw.bits() + '11'


# ---------------------------------------------------------------------------------------------
# reading (checking and analysis of the 360 files)

class BitReader:
    def __init__(self, bits, pos=0):
        self.s = bits; self.pos = pos

    def get(self, k):
        if k == 0: return 0
        v = self.s[self.pos:self.pos + k]
        if len(v) < k: raise EOFError
        self.pos += k
        return int(v, 2)

    def sget(self, k):
        v = self.get(k)
        return v - (1 << k) if v >> (k - 1) else v

    def vlc(self, t):
        dec, ml = t[1], t[2]; c = 0
        for l in range(1, ml + 1):
            c = (c << 1) | self.get(1)
            if (l, c) in dec: return dec[(l, c)]
        raise ValueError('invalid VLC code')

    def large(self):
        n = 8
        if self.get(1):
            n += 8
            if self.get(1):
                n += 8
                if self.get(1): n += 7
        return self.get(n)


def read_coefs(br, sl, tnv, nvc):
    tab = br.get(1)
    co = [0] * sl; cur = 0; rl = 0; nz = 0; thr = sl >> 8
    while (tnv or not rl) and cur + 3 < nvc:
        idx = br.vlc(V4_VLC)
        if idx < 0:
            vals = []
            for i in range(2):
                i2 = br.vlc(V2_VLC)
                if i2 < 0:
                    for j in range(2):
                        v = br.vlc(V1_VLC)
                        if v == 100: v += br.large()
                        vals.append(v)
                else:
                    vals += [i2 >> 4, i2 & 15]
        else:
            vals = [idx >> 12, (idx >> 8) & 15, (idx >> 4) & 15, idx & 15]
        for v in vals:
            if v:
                co[cur] = v if br.get(1) else -v; nz = 0
            else:
                nz += 1
                if nz > thr: rl = 1
            cur += 1
    if cur < sl:
        vlc = COEF_VLC[tab]; run = COEF_RUN[tab]; lev = COEF_LEVEL[tab]
        off = cur; esc_len = (sl - 1).bit_length()
        while off < sl:
            code = br.vlc(vlc)
            if code > 1:
                off += run[code]
                if off >= sl: raise ValueError('run past the subframe')
                co[off] = lev[code] if br.get(1) else -lev[code]
            elif code == 1:
                break
            else:
                level = br.large()
                if br.get(1):
                    if br.get(1):
                        if br.get(1): raise ValueError('invalid escape')
                        off += br.get(esc_len) + 4
                    else:
                        off += br.get(2) + 1
                if off >= sl: raise ValueError('run past the subframe')
                co[off] = level if br.get(1) else -level
            off += 1
    return tab, co


def parse_frame(br, nch, offs, chs):
    """Reads a frame as FFmpeg does. chs: per-channel state (saved scale factors)."""
    start = br.pos
    flen = br.get(15)
    if flen in (0, 0x7FFF):
        return None
    fixed = br.get(1)
    subs_len = [[] for _ in range(nch)]
    num = [0] * nch; minlen = 0; cfc = nch
    while minlen < SPF:
        contains = []
        for c in range(nch):
            if num[c] == minlen:
                contains.append(1 if (fixed or cfc == 1 or minlen == SPF - SUB) else br.get(1))
            else:
                contains.append(0)
        if minlen == SPF - SUB:
            sl = SUB
        else:
            sh = 1 + br.get(1) if br.get(1) else 0
            sl = SPF >> sh
        minlen += sl
        for c in range(nch):
            if contains[c]:
                subs_len[c].append(sl); num[c] += sl
            elif num[c] <= minlen:
                if num[c] < minlen: cfc = 0; minlen = num[c]
                cfc += 1
    if not fixed:
        raise ValueError('different split per channel: not supported')
    fr = dict(tiles=subs_len, start=start, len=flen)
    if nch > 1:
        fr['postproc'] = br.get(1)
        if fr['postproc']: raise ValueError('postproc not supported')
    fr['drc'] = br.get(8)
    ts = te = None
    if br.get(1):
        if br.get(1): ts = br.get(10)
        if br.get(1): te = br.get(10)
    fr['trim'] = (ts, te)
    sfo = sf_resample_table(offs)
    for c in range(nch): chs[c]['reuse'] = 0
    subs = []
    for si, sl in enumerate(subs_len[0]):
        chans = list(range(nch))
        ti = {512: 0, 256: 1, 128: 2}[sl]; nb = len(offs[ti]) - 1
        sub = dict(len=sl, chans=chans)
        if br.get(1): raise ValueError('extended header not supported')
        if br.get(1): raise ValueError('reserved bit')
        sub['tr'] = None
        if nch > 1:
            if br.get(1): raise ValueError('channel transform bit')
            if br.get(1):
                if br.get(1): raise ValueError('unknown transform')
                sub['tr'] = 'none'
            else:
                sub['tr'] = 'ms'
                sub['tb'] = 'all' if br.get(1) else [br.get(1) for _ in range(nb)]
        sub['tc'] = {c: br.get(1) for c in chans}
        if any(sub['tc'].values()):
            sub['tnv'] = br.get(1)
            sub['nvc'] = {}
            if sub['tnv']:
                nbits = ((sl + 3) // 4).bit_length()
                for c in chans: sub['nvc'][c] = br.get(nbits) << 2
            step = br.sget(6); q = 90 + step
            if step in (-32, 31):
                acc = 0
                while True:
                    s5 = br.get(5)
                    if s5 != 31: break
                    acc += 31
                q += (acc + s5) if step == 31 else -(acc + s5)
            sub['q'] = q
            if len(chans) > 1:
                ml = br.get(3); mods = {}
                for c in chans:
                    if br.get(1): mods[c] = (br.get(ml) + 1) if ml else 1
                sub['mod'] = (ml, mods)
            sub['sf'] = {}
            for c in chans:
                ch = chs[c]
                if ch['reuse']:
                    so = sfo[ti][ch['ti']]
                    cur_sf = [ch['saved'][so[b]] for b in range(nb)]
                else:
                    cur_sf = None
                if si == 0 or br.get(1):
                    if not ch['reuse']:
                        stp = br.get(2) + 1; val = 45 // stp; vals = []
                        for b in range(nb):
                            val += br.vlc(SF_VLC); vals.append(val)
                        ch['step'] = stp; cur_sf = vals
                        sub['sf'][c] = ('dpcm', stp, vals)
                    else:
                        diffs = []; i = 0
                        while i < nb:
                            idx = br.vlc(SFRL_VLC)
                            if idx == 0:
                                code = br.get(14); val = code >> 6; sign = code & 1; skip = (code & 0x3F) >> 1
                            elif idx == 1:
                                break
                            else:
                                skip = SCALE_RL_RUN[idx]; val = SCALE_RL_LEVEL[idx]; sign = br.get(1)
                            i += skip
                            if i >= nb: raise ValueError('run-level scale factors')
                            d = val if sign else -val
                            cur_sf[i] += d; diffs.append((i, d)); i += 1
                        sub['sf'][c] = ('rl', diffs)
                    ch['saved'] = list(cur_sf); ch['ti'] = ti; ch['reuse'] = 1
                else:
                    sub['sf'][c] = ('keep',)
                sub.setdefault('sfv', {})[c] = (ch['step'], list(cur_sf))
        sub['coefs'] = {}
        for c in chans:
            if sub['tc'][c]:
                sub['coefs'][c] = read_coefs(br, sl, sub['tnv'], sub['nvc'][c] if sub['tnv'] else sl)
        subs.append(sub)
    fr['subs'] = subs
    used = br.pos - start
    if flen != used + 2:
        raise ValueError('frame length %d differs from the bits read %d' % (flen, used + 2))
    fr['pad'] = br.get(1); fr['more'] = br.get(1)
    return fr


def bits_of(data):
    return ''.join(map('{:08b}'.format, data))


def packet_header(h):
    return h >> 26, (h >> 11) & 0x7FFF, (h >> 8) & 7, h & 0xFF


def parse_stream(data, nch, rate=32000):
    """Reads every block of an XMA2 stream. Returns the list of blocks (lists of frames)."""
    offs = band_offsets(rate)
    npk = len(data) // PKT
    blocks = []
    chs = [dict(reuse=0, saved=None, ti=0, step=1) for _ in range(nch)]
    for b0 in range(0, npk, BLOCK_PACKETS):
        pk = range(b0, min(npk, b0 + BLOCK_PACKETS))
        bits = ''.join(bits_of(data[k * PKT + 4:(k + 1) * PKT]) for k in pk)
        first = packet_header(struct.unpack_from('>I', data, b0 * PKT)[0])[1]
        br = BitReader(bits, first)
        frames = []
        while br.pos + 15 <= len(bits):
            try:
                f = parse_frame(br, nch, offs, chs)
            except EOFError:
                break
            if f is None:
                # padding: the next frame (if any) starts in a following packet
                k = (br.pos - 15) // PB + 1; found = False
                while k < len(pk):
                    h = packet_header(struct.unpack_from('>I', data, pk[k] * PKT)[0])
                    if h[0]:
                        br.pos = k * PB + h[1]; found = True; break
                    k += 1
                if not found: break
                continue
            frames.append(f)
        blocks.append(frames)
    return blocks


# ---------------------------------------------------------------------------------------------
# packing

def pack(frames_bits):
    """frames_bits: frame bit strings (last bit = "another frame", recomputed here).
    Rules of the 360 game: frames are contiguous also across packets; at most 63 frames
    starting in the same packet (the field is 6 bits), beyond that the rest of the packet is filled;
    a frame does not cross the boundary of a 16-packet block.
    Returns (data, number of frames per block)."""
    blocks = []
    cur = []; pos = 0
    for fb in frames_bits:
        assert len(fb) < PB
        k = pos // PB
        if sum(1 for s, _ in cur if s // PB == k) >= 63:
            pos = (k + 1) * PB
        if pos + len(fb) > BLOCK_PACKETS * PB:
            blocks.append(cur); cur = []; pos = 0
        cur.append((pos, fb)); pos += len(fb)
    if cur: blocks.append(cur)
    out = bytearray(); counts = []
    for bl in blocks:
        end = bl[-1][0] + len(bl[-1][1])
        npk = (end + PB - 1) // PB
        bits = bytearray(b'1' * (npk * PB))
        for i, (st, fb) in enumerate(bl):
            nxt = bl[i + 1][0] if i + 1 < len(bl) else None
            more = 1 if nxt is not None and nxt // PB == st // PB else 0
            bits[st:st + len(fb)] = (fb[:-1] + str(more)).encode()
        bits = bits.decode()
        starts = [st for st, _ in bl]
        for k in range(npk):
            inpk = [s - k * PB for s in starts if k * PB <= s < (k + 1) * PB]
            if inpk:
                h = (len(inpk) << 26) | (inpk[0] << 11) | (1 << 8)
            else:
                h = 0x7FFF << 11
            out += struct.pack('>I', h)
            out += int(bits[k * PB:(k + 1) * PB], 2).to_bytes(PB // 8, 'big')
        counts.append(len(bl))
    return bytes(out), counts


def seek_chunk(counts):
    """FSB chunk 6: 1, number of blocks + 2, then 0 and the decoded samples at the end of each
    block (frames * 512), big endian."""
    vals = [1, len(counts) + 2, 0]; acc = 0
    for c in counts:
        acc += c * SPF; vals.append(acc)
    return struct.pack('>%dI' % len(vals), *vals)


# ---------------------------------------------------------------------------------------------
# encoding

_N = SUB
_j = np.arange(2 * _N)
WINDOW = np.sin((_j + 0.5) * np.pi / (2 * _N))
# MDCT as in ff_tx_mdct_naive_fwd, with the sign the decoder expects (checked: output = x/32768)
MDCT = -np.cos(np.pi / (4 * _N) * np.outer(2 * np.arange(_N) + 1, 2 * _j + 1 + _N)) * WINDOW[None, :]


def analyze(x, nframes):
    """x: signal of a channel already positioned (length >= nframes*512+512).
    Returns an array (frame, subframe, 128) of coefficients in decoder units."""
    starts = (np.arange(nframes)[:, None] * SPF + np.arange(4)[None, :] * SUB + 192).ravel()
    idx = starts[:, None] + _j[None, :]
    seg = x[idx]                                 # (nframes*4, 256)
    return (seg @ MDCT.T).reshape(nframes, 4, _N)


class Params:
    def __init__(self, noise=0.6, tilt=0.6, floor_db=40.0, sf_step=2, deadzone=0.15, ms=True, vec_len=False):
        self.noise = noise          # quantization step relative to the band energy
        self.tilt = tilt            # 1 = constant SNR per band, <1 = more noise in weak bands
        self.floor_db = floor_db    # absolute minimum step (dB in decoder units)
        self.sf_step = sf_step
        self.deadzone = deadzone
        self.ms = ms                # stereo: mid/side when the channels are similar
        self.vec_len = vec_len      # length of the transmitted vector part (the 360 game does not use it)


def _band_rms(X, offs):
    # X (..., 128) -> (..., nb)
    return np.stack([np.sqrt(np.mean(X[..., a:b] ** 2, axis=-1) + 1e-9) for a, b in zip(offs[:-1], offs[1:])], -1)


def _quant(X, step, dz):
    return (np.sign(X) * np.floor(np.abs(X) / step + 0.5 - dz)).astype(np.int64)


def _coef_options(q, vec_len):
    """Bits needed for a channel in the two modes (tnv 0/1): {tnv: (bits, table, nvc)}."""
    last = max((i for i, v in enumerate(q) if v), default=-1)
    res = {}
    modes = [(0, (SUB,))]
    if vec_len:
        modes.append((1, sorted({0, min(SUB, (last + 4) // 4 * 4)})))
    for tnv, nvcs in modes:
        for nvc in nvcs:
            for tab in (0, 1):
                bw = BitWriter(); write_coefs(bw, q, tab, tnv, nvc)
                if tnv not in res or bw.n < res[tnv][0]:
                    res[tnv] = (bw.n, tab, nvc)
    return res


def encode_frames(pcm, prm, loop=False, rate=32000):
    """pcm: float array (samples, channels) in int16 units, length a multiple of 512.
    Returns the list of frame bit strings."""
    n, nch = pcm.shape
    assert n % SPF == 0 and nch in (1, 2)
    nframes = n // SPF + 1
    all_offs = band_offsets(rate)
    total = nframes * SPF + SPF
    X = []
    for c in range(nch):
        x = np.zeros(total)
        x[SPF:SPF + n] = pcm[:, c]
        if loop:      # circular continuity: the sound itself comes before and after the sound
            x[:SPF] = pcm[-SPF:, c]
            x[SPF + n:] = np.resize(pcm[:, c], total - SPF - n)
        X.append(analyze(x, nframes))
    X = np.stack(X)                              # (channel, frame, subframe, 128)
    frames = []
    for f in range(nframes):
        # a frame must fit in a packet: if it comes out too large it is quantized more coarsely
        mult = 1.0
        while True:
            fb = _encode_frame(X[:, f], f == 0, f == nframes - 1, prm, mult, all_offs)
            if len(fb) < MAX_FRAME_BITS:
                break
            mult *= 1.4
        frames.append(fb)
    return frames


MAX_FRAME_BITS = 15000


def _encode_frame(Xf, first, last, prm, mult, all_offs):
    """One frame (4 subframes of 128) of one or two channels -> bit string."""
    nch = Xf.shape[0]
    offs = all_offs[2]
    width = np.diff(offs)
    floor = 10 ** (prm.floor_db / 20)
    step = prm.sf_step
    fr = dict(subs=[], trim=(SPF if first else None, SUB if last else None))
    C = Xf; use_ms = False
    if nch == 2 and prm.ms:
        M = (Xf[0] + Xf[1]) / 2; S = (Xf[1] - Xf[0]) / 2
        eL = np.sum(Xf[0] ** 2); eR = np.sum(Xf[1] ** 2); eS = np.sum(S ** 2)
        if eS < 0.25 * min(eL, eR) or eS < 0.1 * max(eL, eR):
            use_ms = True; C = np.stack([M, S])
    # spectral shape of the frame (rms per band over the 4 subframes) and desired step per band
    shape = np.sqrt(np.mean(_band_rms(Xf, offs) ** 2, axis=1))      # (channel, band)
    peak = np.max(shape, axis=1, keepdims=True)
    des = np.maximum(prm.noise * mult * shape ** prm.tilt * peak ** (1 - prm.tilt), floor)
    sub_rms = np.sqrt(np.mean(Xf ** 2, axis=-1) + 1e-9)             # (channel, subframe)
    frm_rms = np.sqrt(np.mean(sub_rms ** 2, axis=1, keepdims=True))
    lvl = np.minimum(20 * np.log10(np.maximum(sub_rms / frm_rms, 1e-3)) * prm.tilt, 0)
    if use_ms:
        des = np.repeat(np.min(des, axis=0, keepdims=True) / np.sqrt(2), 2, 0)
        lvl = np.repeat(np.max(lvl, axis=0, keepdims=True), 2, 0)
    u = np.round(20 * np.log10(des) / step).astype(int)            # scale factors
    umax = u.max(axis=1)
    u = np.maximum(u, umax[:, None] - 55)       # DPCM differences within the +-60 of the code
    sfvals = [[int(v - umax[c] + 45 // step) for v in u[c]] for c in range(nch)]
    reuse = [0] * nch
    for s in range(4):
        sub = dict(len=SUB, chans=list(range(nch)), tc={}, sf={}, coefs={}, nvc={})
        Qc = [int(round(umax[c] * step + lvl[c, s])) for c in range(nch)]
        if nch == 2:
            sub['tr'] = 'ms' if use_ms else 'none'; sub['tb'] = 'all'
            base = min(Qc)
            Qc = [min(q, base + 128) for q in Qc]
        qs = []
        for c in range(nch):
            E = u[c] * step + (Qc[c] - umax[c] * step)
            qs.append([int(v) for v in _quant(C[c, s], np.repeat(10 ** (E / 20), width), prm.deadzone)])
            sub['tc'][c] = 1 if any(qs[c]) else 0
        if any(sub['tc'].values()):
            sub['q'] = min(Qc)
            if nch == 2:
                mods = {c: Qc[c] - sub['q'] for c in range(2) if Qc[c] > sub['q']}
                ml = max((m - 1).bit_length() for m in mods.values()) if mods else 0
                sub['mod'] = (ml, mods)
            for c in range(nch):
                if s == 0 or not reuse[c]:
                    sub['sf'][c] = ('dpcm', step, sfvals[c]); reuse[c] = 1
                else:
                    sub['sf'][c] = ('keep',)
            opts = {c: _coef_options(qs[c], prm.vec_len) for c in range(nch) if sub['tc'][c]}
            cost = {t: sum(o[t][0] for o in opts.values()) + (6 * nch if t else 0)
                    for t in ((0, 1) if prm.vec_len else (0,))}
            tnv = min(cost, key=cost.get)
            sub['tnv'] = tnv
            for c, o in opts.items():
                sub['coefs'][c] = (o[tnv][1], qs[c]); sub['nvc'][c] = o[tnv][2]
            for c in range(nch):
                sub['nvc'].setdefault(c, 0)
        fr['subs'].append(sub)
    return write_frame(fr, nch, all_offs)


def encode(pcm, prm=None, loop=False):
    """PCM (samples, channels) multiple of 512 -> (XMA2 data, chunk 6, samples)."""
    prm = prm or Params()
    fb = encode_frames(pcm, prm, loop)
    data, counts = pack(fb)
    return data, seek_chunk(counts), len(pcm)


RATE = 32000      # rate of every XMA sound of the 360 game (index 7 in the FSB5 header)


def prepare_pcm(pcm, src_rate, loop=False):
    """int16 (samples, channels) at src_rate -> float at 32000 Hz with a length multiple of 512.

    Looping sounds: periodic FFT resampling to the nearest length multiple of 512 (the pitch
    changes by a few thousandths), so the loop point stays continuous. The others: polyphase filter
    and zero tail up to the whole frame."""
    from scipy import signal
    from math import gcd
    x = pcm.astype(np.float64)
    if loop:
        m = max(SPF, int(round(len(x) * RATE / src_rate / SPF)) * SPF)
        return signal.resample(x, m, axis=0)
    if src_rate != RATE:
        g = gcd(src_rate, RATE)
        x = signal.resample_poly(x, RATE // g, src_rate // g, axis=0)
    return np.concatenate([x, np.zeros(((-len(x)) % SPF, x.shape[1]))])


def fsb_names_360(names):
    """Name table of a Wii U FSB ("x.wav,0,28000,kIma4,40,1,0,0") in the form of the 360 game
    ("x.wav,0,32000,kFsbXma,40,1,0,0"). Returns the string without padding."""
    o = struct.unpack_from('<I', names, 0)[0]
    txt = names[o:names.index(b'\0', o)].decode('latin1').split(',')
    if len(txt) >= 4:
        txt[2] = str(RATE); txt[3] = 'kFsbXma'
    return struct.pack('<I', 4) + ','.join(txt).encode('latin1') + b'\0'


def to_fsb(f, data, chunk6, samples, channels):
    """XMA FSB5 from a parsed FSB (fsb.parse): same header and hash, converted name, 360 fields:
    codec 10, rate index 7, only chunk 6, name table padded to a multiple of 32."""
    import fsb
    g = dict(f)
    g['codec'] = fsb.CODEC_XMA; g['freqidx'] = 7; g['channels'] = channels
    g['chunks'] = [(6, chunk6)]; g['samples'] = samples; g['offset'] = 0; g['data'] = data
    names = fsb_names_360(f['names']) if b'kIma4' in f['names'] or b'kFsbXma' not in f['names'] else \
        f['names'].rstrip(b'\0') + b'\0'
    shs = 8 + 4 + len(chunk6)
    names += b'\0' * ((-(0x3C + shs + len(names))) % 32)
    g['names'] = names
    return fsb.build(g)


# ---------------------------------------------------------------------------------------------
# containers

def riff_xma2(data, channels, rate, samples, path):
    """XMA2 WAV (format 0x166) for decoding with ffmpeg."""
    nblocks = (len(data) + 32767) // 32768
    ext = struct.pack('<HIIIIIIIBBH', 1, 0 if channels == 1 else 3, samples + SPF, 32768, 0, samples,
                      0, 0, 0, 4, nblocks)
    fmt = struct.pack('<HHIIHHH', 0x166, channels, rate, rate * channels * 2, 2048, 16, len(ext)) + ext
    body = b'WAVE' + b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'data' + struct.pack('<I', len(data)) + data
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', len(body)) + body)


def ffmpeg_decode(data, channels, samples, work, rate=RATE):
    """Decodes with ffmpeg (xma2 decoder). Returns (float pcm (samples, channels), errors).
    FFmpeg drops the first 64 samples of the XMA output. The muxer message about non increasing dts
    is not a decoding error: it also shows up with files of the 360 game when the last packet holds
    only the tail of the last frame, and is ignored."""
    import subprocess
    wav = os.path.join(work, 'xma_check.wav'); raw = os.path.join(work, 'xma_check.raw')
    riff_xma2(data, channels, rate, samples, wav)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', wav, '-f', 'f32le',
                        '-acodec', 'pcm_f32le', raw], capture_output=True, text=True)
    err = [l for l in r.stderr.splitlines() if l.strip() and 'non monotonically increasing dts' not in l]
    y = np.fromfile(raw, dtype='<f4').reshape(-1, channels)
    return y, '\n'.join(err)


def snr_db(pcm, y):
    """SNR between the PCM given to the encoder (int16 units) and the ffmpeg output."""
    ref = pcm[64:] / 32768.0; n = min(len(ref), len(y))
    return 10 * np.log10(np.sum(ref[:n] ** 2) / max(np.sum((ref[:n] - y[:n]) ** 2), 1e-20) + 1e-20)


def check_stream(data, channels, rate=RATE):
    """Re-reads a stream and rewrites it: True if the bytes match (parser and serializer agree)."""
    blocks = parse_stream(data, channels, rate)
    offs = band_offsets(rate)
    out = b''
    for bl in blocks:
        out += pack([write_frame(fr, channels, offs) for fr in bl])[0]
    return out == data, blocks


if __name__ == '__main__':
    # python3 xma.py riff file.fsb output.wav  : wraps the XMA of an FSB in an XMA2 WAV (for ffmpeg)
    # python3 xma.py check file.fsb ...        : identical re-read and rewrite of the stream
    import fsb
    cmd = sys.argv[1]
    if cmd == 'riff':
        f = fsb.parse(open(sys.argv[2], 'rb').read())
        riff_xma2(f['data'], f['channels'], f['rate'], f['samples'], sys.argv[3])
    elif cmd == 'check':
        for p in sys.argv[2:]:
            f = fsb.parse(open(p, 'rb').read())
            ok, blocks = check_stream(f['data'], f['channels'], f['rate'])
            print('%s: %s, %d blocks, %d frames' % (p, 'identical' if ok else 'DIFFERENT', len(blocks),
                                                    sum(len(b) for b in blocks)))
