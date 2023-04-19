((0 uniq "" ((1 ins 6))
  ((56 nul 11) (71 nul 11) (84 nul 11) (91 nul 11) (92 nul 11) (96 nul 11)
   (97 nul 11) (188 nul 13) (190 nul 13) (222 nul 13) (225 nul 13)
   (228 nul 13) (229 nul 13) (230 nul 13) (231 nul 13) (247 nul 13)
   (248 nul 13) (338 nul 14) (340 nul 14) (385 nul 14) (602 nul 16)
   (609 nul 16) (630 nul 16) (664 nul 16) (708 nul 17) (890 nul 18)
   (894 nul 18) (934 nul 18) (983 nul 18) (990 nul 19) (1268 nul 20)
   (1533 nul 23))
  -1 0)
 (1 uniq "move ua + 4 , ul "
  ((0 del 6) (5 ins 2) (6 ins 2) (7 ins 2) (8 ins 2) (9 ins 2) (10 ins 2))
  ((7 ins 2) (8 ins 2) (9 ins 2) (65 nul 7) (121 nul 7) (130 nul 10)
   (151 nul 7) (153 nul 9) (202 nul 9) (210 nul 10) (220 nul 7) (223 nul 7)
   (285 nul 12) (322 nul 11) (346 nul 9) (353 nul 11) (451 nul 11)
   (555 nul 12) (580 nul 12) (637 nul 14) (653 nul 11) (744 nul 15)
   (762 nul 12) (810 nul 14) (823 nul 12) (842 nul 15) (1066 nul 17)
   (1099 nul 16) (1200 nul 15) (1257 nul 15) (1335 nul 18) (1419 nul 18))
  -1 0)
 (2 uniq "move ua + 4 , 2 - 3 "
  ((3 sub 2) (4 sub 2) (5 sub 2) (6 sub 2) (7 sub 1) (8 sub 3) (9 sub 2)
   (10 sub 2))
  ((3 sub 2) (5 sub 2) (6 sub 2) (10 sub 2) (129 nul 10) (185 nul 7)
   (211 nul 11) (224 nul 7) (344 nul 10) (368 nul 8) (391 nul 9) (410 nul 9)
   (739 nul 15) (837 nul 14) (1173 nul 16) (1232 nul 16) (1309 nul 17)
   (1514 nul 19))
  -1 0)
 (3 equiv "move ua + 4 , 3 - 4 "
  ((2 sub 2) (4 sub 3) (5 sub 2) (6 sub 2) (7 sub 2) (8 sub 3) (9 sub 2)
   (10 sub 2))
  () 2 0)
 (4 uniq "move ua + 4 , 4 * 3 "
  ((2 sub 2) (3 sub 3) (5 sub 3) (6 sub 3) (7 sub 3) (8 sub 3) (9 sub 3)
   (10 sub 3))
  ((327 nul 11)) -1 0)
 (5 equiv "move ua + 4 , 0 - ul "
  ((1 del 2) (2 sub 2) (3 sub 2) (4 sub 3) (6 sub 2) (7 sub 1) (8 sub 2)
   (9 sub 2) (10 sub 1))
  () 2 0)
 (6 equiv "move ua + 4 , ul - 2 "
  ((1 del 2) (2 sub 2) (3 sub 2) (4 sub 3) (5 sub 2) (7 sub 2) (8 sub 3)
   (9 sub 1) (10 sub 2))
  () 2 0)
 (7 equiv "move ua + 4 , 2 - ul "
  ((1 del 2) (2 sub 1) (3 sub 2) (4 sub 3) (5 sub 1) (6 sub 2) (8 sub 2)
   (9 sub 2) (10 sub 1))
  () 1 0)
 (8 equiv "move ua + 4 , ua + ul "
  ((1 del 2) (2 sub 3) (3 sub 3) (4 sub 3) (5 sub 2) (6 sub 3) (7 sub 2)
   (9 sub 3) (10 sub 1))
  () 1 0)
 (9 equiv "move ua + 4 , ul - ua "
  ((1 del 2) (2 sub 2) (3 sub 2) (4 sub 3) (5 sub 2) (6 sub 1) (7 sub 2)
   (8 sub 3) (10 sub 2))
  () 1 0)
 (10 equiv "move ua + 4 , ua - ul "
  ((1 del 2) (2 sub 2) (3 sub 2) (4 sub 3) (5 sub 1) (6 sub 2) (7 sub 1)
   (8 sub 1) (9 sub 2))
  () 2 0)
 (11 uniq "( pen 3 ; move 4 , 2 ) "
  ((12 sub 2) (13 sub 2) (14 sub 3) (15 sub 2) (16 sub 2) (17 sub 3)
   (18 sub 3) (93 ins 3) (94 ins 3) (109 ins 4) (238 ins 5))
  ((24 nul 3) (57 nul 3) (93 ins 3) (94 ins 3) (109 ins 4) (238 ins 5)
   (254 nul 6) (272 nul 6))
  -1 0)
 (12 uniq "( pen ua ; move 4 , 0 ) "
  ((11 sub 2) (13 sub 3) (14 sub 1) (15 sub 1) (16 sub 1) (17 sub 3)
   (18 sub 2) (22 ins 2) (23 ins 2) (46 ins 2) (99 ins 3) (100 ins 3)
   (135 ins 4) (136 ins 4) (246 ins 5) (257 ins 5))
  ((15 sub 1) (22 ins 2) (23 ins 2) (28 nul 4) (29 nul 4) (46 ins 2)
   (88 nul 3) (99 ins 3) (100 ins 3) (110 nul 4) (111 nul 4) (186 nul 6)
   (214 nul 6) (257 ins 5) (316 nul 7) (465 nul 7) (490 nul 7) (501 nul 7)
   (503 nul 8) (578 nul 8) (620 nul 7) (683 nul 9) (716 nul 9) (728 nul 10)
   (838 nul 11) (925 nul 12) (1109 nul 13) (1214 nul 12) (1325 nul 14)
   (1334 nul 13) (1372 nul 14) (1500 nul 15))
  -1 0)
 (13 uniq "( pen 3 ; move ua , 4 ) "
  ((11 sub 2) (12 sub 3) (14 sub 2) (15 sub 3) (16 sub 3) (17 sub 2)
   (18 sub 2) (52 ins 2) (141 ins 4) (147 ins 4) (250 ins 5) (263 ins 5)
   (268 ins 5) (270 ins 5) (273 ins 5) (300 ins 5))
  ((48 nul 3) (49 nul 3) (180 nul 6) (250 ins 5) (270 ins 5) (337 nul 7)
   (573 nul 8) (859 nul 10) (1035 nul 10) (1123 nul 12))
  -1 0)
 (14 uniq "( pen ua ; move ua , 0 ) "
  ((11 sub 3) (12 sub 1) (13 sub 2) (15 sub 2) (16 sub 2) (17 sub 2)
   (18 sub 1) (68 ins 2) (71 ins 2) (98 ins 3) (276 ins 5) (280 ins 5)
   (293 ins 5) (338 ins 5) (420 ins 6))
  ((18 sub 1) (68 ins 2) (74 nul 3) (75 nul 3) (79 nul 3) (80 nul 4)
   (95 nul 3) (98 ins 3) (221 nul 5) (280 ins 5) (293 ins 5) (420 ins 6)
   (421 nul 8) (423 nul 8) (504 nul 7) (511 nul 8) (512 nul 10) (542 nul 9)
   (561 nul 7) (568 nul 9) (685 nul 9) (706 nul 9) (714 nul 11) (775 nul 10)
   (872 nul 10) (979 nul 12) (982 nul 9) (997 nul 9) (1302 nul 13)
   (1330 nul 12) (1374 nul 14) (1375 nul 15))
  -1 0)
 (15 equiv "( pen ua ; move 4 , ua ) "
  ((11 sub 2) (12 sub 1) (13 sub 3) (14 sub 2) (16 sub 1) (17 sub 3)
   (18 sub 1) (56 ins 2) (88 ins 2) (110 ins 3) (111 ins 3) (146 ins 4)
   (187 ins 4) (188 ins 4) (321 ins 5))
  () 12 0)
 (16 uniq "( pen ua ; move 4 , ul ) "
  ((11 sub 2) (12 sub 1) (13 sub 3) (14 sub 2) (15 sub 1) (17 sub 2)
   (18 sub 2) (54 ins 2) (55 ins 2) (59 ins 2) (60 ins 2) (77 ins 2)
   (89 ins 2) (92 ins 2) (100 ins 3) (111 ins 3) (187 ins 4) (217 ins 4)
   (269 ins 5) (296 ins 5) (297 ins 5) (308 ins 5) (310 ins 5) (328 ins 5)
   (339 ins 5))
  ((30 nul 3) (47 nul 3) (54 ins 2) (55 ins 2) (58 nul 3) (60 ins 2)
   (78 nul 3) (89 ins 2) (139 nul 6) (187 ins 4) (246 nul 6) (269 ins 5)
   (279 nul 6) (308 ins 5) (310 ins 5) (425 nul 7) (458 nul 7) (545 nul 8)
   (554 nul 7) (560 nul 8) (570 nul 8) (577 nul 7) (631 nul 7) (666 nul 10)
   (772 nul 10) (785 nul 10) (827 nul 9) (1020 nul 10) (1146 nul 11)
   (1147 nul 12) (1278 nul 12) (1452 nul 14))
  -1 0)
 (17 uniq "( pen 4 ; move ua , ul ) "
  ((11 sub 3) (12 sub 3) (13 sub 2) (14 sub 2) (15 sub 3) (16 sub 2)
   (18 sub 2) (81 ins 2) (90 ins 2) (184 ins 4) (256 ins 5) (259 ins 5)
   (261 ins 5) (264 ins 5) (271 ins 5) (275 ins 5) (292 ins 5) (295 ins 5)
   (298 ins 5) (299 ins 5) (302 ins 5) (304 ins 5) (309 ins 5) (323 ins 5))
  ((50 nul 3) (81 ins 2) (82 nul 3) (90 ins 2) (181 nul 5) (219 nul 5)
   (256 ins 5) (261 ins 5) (266 nul 6) (271 ins 5) (275 ins 5) (284 nul 6)
   (304 ins 5) (306 nul 6) (309 ins 5) (383 nul 7) (476 nul 7) (546 nul 8)
   (636 nul 7) (684 nul 10) (826 nul 9) (888 nul 10) (913 nul 9)
   (1038 nul 11) (1186 nul 12) (1274 nul 12) (1328 nul 13) (1339 nul 13)
   (1340 nul 14) (1395 nul 14))
  -1 0)
 (18 equiv "( pen ua ; move ua , ua ) "
  ((11 sub 3) (12 sub 2) (13 sub 2) (14 sub 1) (15 sub 1) (16 sub 2)
   (17 sub 2) (69 ins 2) (74 ins 2) (75 ins 2) (79 ins 2) (95 ins 2)
   (96 ins 2) (179 ins 4) (182 ins 4) (215 ins 4) (216 ins 4) (218 ins 4)
   (221 ins 4) (222 ins 4) (225 ins 4) (226 ins 4) (247 ins 4) (276 ins 5)
   (280 ins 5) (293 ins 5) (316 ins 5) (320 ins 5) (321 ins 5) (385 ins 5))
  () 14 0)
 (19 uniq "( pen 3 ; move 4 + 2 , 0 ) " ((47 sub 3) (69 sub 3))
  ((244 nul 4) (245 nul 5) (462 nul 7) (521 nul 8) (1366 nul 14)) -1 0)
 (20 uniq "( move 4 , 0 ; move ua , 2 ) "
  ((32 sub 1) (33 sub 2) (34 sub 2) (35 sub 3) (36 sub 3) (37 sub 1)
   (38 sub 3) (39 sub 3) (40 sub 2) (41 sub 3) (44 sub 2) (61 sub 2)
   (62 sub 3) (63 sub 3) (64 sub 3) (101 ins 2) (116 ins 2) (345 ins 4))
  ((37 sub 1) (116 ins 2) (117 nul 4) (158 nul 4) (349 nul 6) (494 nul 6)
   (502 nul 6) (510 nul 6) (695 nul 9) (700 nul 8))
  -1 0)
 (21 uniq "( move ua , 2 ; move 4 , 4 ) "
  ((22 sub 3) (32 sub 3) (33 sub 2) (35 sub 2) (36 sub 3) (38 sub 2)
   (39 sub 2) (40 sub 3) (41 sub 2) (42 sub 2) (43 sub 1) (44 sub 3)
   (45 sub 2) (54 sub 3) (61 sub 3) (62 sub 2) (63 sub 2) (64 sub 2)
   (65 sub 2) (66 sub 2) (67 sub 2) (106 ins 2) (107 ins 2) (361 ins 4)
   (456 ins 5))
  ((107 ins 2) (777 nul 7) (1034 nul 11) (1063 nul 9)) -1 0)
 (22 equiv "( pen ua + 2 ; move 4 , 0 ) "
  ((12 del 2) (21 sub 3) (23 sub 1) (24 sub 3) (30 sub 3) (46 sub 1)
   (54 sub 2) (55 sub 2) (58 sub 3) (60 sub 3) (68 sub 3) (75 sub 3)
   (88 sub 3) (89 sub 2))
  () 12 0)
 (23 equiv "( pen ua + 4 ; move 4 , 0 ) "
  ((12 del 2) (22 sub 1) (24 sub 3) (45 sub 3) (46 sub 1) (54 sub 3)
   (55 sub 1) (58 sub 3) (60 sub 2) (68 sub 3) (88 sub 3) (89 sub 2))
  () 12 0)
 (24 equiv "( pen 2 + ul ; move 4 , 2 ) "
  ((22 sub 3) (23 sub 3) (38 sub 3) (46 sub 2) (48 sub 2) (55 sub 3)
   (56 sub 3) (58 sub 2) (80 sub 3) (82 sub 3) (88 sub 3) (89 sub 2)
   (521 ins 5))
  () 11 0)
 (25 uniq "( pen 4 ; move ua , 4 + 2 ) "
  ((26 sub 2) (31 sub 3) (49 sub 3) (51 sub 2) (74 sub 3) (81 sub 2)
   (90 sub 3) (95 sub 3))
  ((725 nul 7) (1036 nul 11)) -1 0)
 (26 uniq "( pen 2 ; move ua , 4 + 4 ) "
  ((25 sub 2) (31 sub 3) (49 sub 3) (51 sub 3) (52 sub 3) (53 sub 2)
   (81 sub 3) (87 sub 3) (95 sub 3))
  ((986 nul 8)) -1 0)
 (27 uniq "( pen ua ; move 4 , 3 - 4 ) "
  ((31 sub 3) (52 sub 3) (53 sub 3) (59 sub 3) (76 sub 3) (87 sub 2)
   (432 ins 5))
  ((538 nul 7) (779 nul 10) (790 nul 9) (821 nul 9)) -1 0)
 (28 equiv "( pen 3 * 3 ; move 4 , ua ) "
  ((29 sub 1) (30 sub 3) (54 sub 3) (56 sub 3) (60 sub 3) (75 sub 3)
   (79 sub 2) (88 sub 3))
  () 12 0)
 (29 equiv "( pen 4 * 3 ; move 4 , ua ) "
  ((28 sub 1) (30 sub 2) (54 sub 3) (56 sub 3) (58 sub 3) (60 sub 3)
   (75 sub 3) (79 sub 2) (80 sub 3) (88 sub 3))
  () 12 0)
 (30 equiv "( pen 4 * 2 ; move 4 , ul ) "
  ((22 sub 3) (28 sub 3) (29 sub 2) (40 sub 3) (50 sub 2) (54 sub 1)
   (55 sub 3) (58 sub 2) (60 sub 2) (75 sub 3) (89 sub 3) (92 sub 3))
  () 16 0)
 (31 uniq "( pen 4 ; move ua , 3 / 4 ) "
  ((25 sub 3) (26 sub 3) (27 sub 3) (49 sub 3) (52 sub 3) (53 sub 3)
   (81 sub 3) (87 sub 3) (90 sub 3))
  () -1 0)
 (32 uniq "( move ua , 0 ; move ua , 2 ) "
  ((20 sub 1) (21 sub 3) (33 sub 1) (34 sub 2) (35 sub 2) (36 sub 2)
   (37 sub 2) (38 sub 2) (39 sub 2) (40 sub 3) (41 sub 2) (42 sub 3)
   (43 sub 3) (44 sub 3) (45 sub 3) (61 sub 1) (62 sub 2) (63 sub 2)
   (64 sub 2) (65 sub 3) (66 sub 3) (67 sub 3) (112 ins 2) (113 ins 2)
   (148 ins 2) (388 ins 4))
  ((61 sub 1) (148 ins 2) (236 nul 4) (274 nul 4) (314 nul 4) (364 nul 5)
   (429 nul 5) (635 nul 6) (668 nul 7) (669 nul 7) (733 nul 7) (735 nul 10)
   (922 nul 10) (1011 nul 9) (1471 nul 13) (1531 nul 13))
  -1 0)
 (33 uniq "( move ua , 0 ; move ua , 4 ) "
  ((20 sub 2) (21 sub 2) (32 sub 1) (34 sub 3) (35 sub 2) (36 sub 2)
   (37 sub 3) (38 sub 3) (39 sub 1) (40 sub 3) (41 sub 1) (42 sub 3)
   (43 sub 2) (44 sub 2) (45 sub 3) (61 sub 2) (62 sub 2) (63 sub 1)
   (64 sub 1) (65 sub 3) (66 sub 3) (67 sub 3) (123 ins 2) (150 ins 2)
   (165 ins 2) (276 ins 3) (565 ins 5))
  ((63 sub 1) (113 nul 3) (194 nul 3) (199 nul 3) (552 nul 6) (565 ins 5)
   (566 nul 6) (584 nul 7) (647 nul 6) (800 nul 8) (843 nul 8) (929 nul 10)
   (1195 nul 10) (1289 nul 11))
  -1 0)
 (34 uniq "( move 3 , ua ; move ua , 2 ) "
  ((20 sub 2) (32 sub 2) (33 sub 3) (36 sub 3) (37 sub 1) (38 sub 3)
   (39 sub 3) (40 sub 3) (41 sub 3) (44 sub 3) (61 sub 1) (62 sub 3)
   (63 sub 2) (64 sub 3) (66 sub 3) (84 sub 3) (158 ins 2) (499 ins 5)
   (526 ins 5) (548 ins 5) (581 ins 5) (679 ins 6))
  ((548 ins 5) (588 nul 7) (638 nul 7) (652 nul 6) (852 nul 10)) -1 0)
 (35 uniq "( move ua , 0 ; move 4 , ul ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 2) (36 sub 3) (38 sub 2)
   (39 sub 3) (40 sub 3) (41 sub 3) (42 sub 1) (43 sub 2) (45 sub 1)
   (54 sub 3) (55 sub 3) (60 sub 3) (61 sub 3) (62 sub 2) (63 sub 3)
   (64 sub 3) (65 sub 2) (66 sub 1) (67 sub 1) (89 sub 3) (155 ins 2)
   (500 ins 5) (505 ins 5) (531 ins 5) (559 ins 5) (592 ins 5) (594 ins 5))
  ((66 sub 1) (154 nul 3) (155 ins 2) (162 nul 4) (200 nul 3) (201 nul 4)
   (282 nul 4) (505 ins 5) (571 nul 6) (594 ins 5) (742 nul 7) (806 nul 9)
   (991 nul 9) (1153 nul 10) (1174 nul 10) (1384 nul 12) (1444 nul 13)
   (1479 nul 13) (1535 nul 14))
  -1 0)
 (36 uniq "( move ua , 3 ; move ua , 3 ) "
  ((20 sub 3) (21 sub 3) (32 sub 2) (33 sub 2) (34 sub 3) (35 sub 3)
   (37 sub 3) (38 sub 3) (39 sub 1) (40 sub 3) (41 sub 2) (42 sub 2)
   (43 sub 3) (44 sub 3) (45 sub 3) (61 sub 2) (62 sub 2) (63 sub 2)
   (64 sub 2) (65 sub 3) (66 sub 3) (67 sub 3) (79 sub 3) (354 ins 4)
   (527 ins 5) (676 ins 6))
  ((171 nul 3) (527 ins 5) (1058 nul 11) (1097 nul 11) (1231 nul 10)) -1 0)
 (37 equiv "( move 4 , ua ; move ua , 2 ) "
  ((20 sub 1) (32 sub 2) (33 sub 3) (34 sub 1) (36 sub 3) (38 sub 3)
   (39 sub 3) (40 sub 2) (41 sub 3) (44 sub 2) (61 sub 1) (62 sub 3)
   (63 sub 2) (64 sub 3) (66 sub 3) (84 sub 3) (124 ins 2) (286 ins 3)
   (348 ins 4) (374 ins 4) (404 ins 4) (498 ins 5) (502 ins 5) (510 ins 5))
  () 20 0)
 (38 uniq "( move ua , ul ; move 4 , 2 ) "
  ((20 sub 3) (21 sub 2) (24 sub 3) (32 sub 2) (33 sub 3) (34 sub 3)
   (35 sub 2) (36 sub 3) (37 sub 3) (39 sub 3) (41 sub 3) (42 sub 2)
   (43 sub 1) (44 sub 3) (45 sub 2) (46 sub 3) (61 sub 2) (62 sub 3)
   (63 sub 3) (64 sub 2) (65 sub 1) (66 sub 2) (67 sub 1) (88 sub 3)
   (89 sub 3) (167 ins 2) (506 ins 5) (661 ins 6) (672 ins 6))
  ((167 ins 2) (450 nul 5) (506 ins 5) (559 nul 6) (749 nul 8) (805 nul 8))
  -1 0)
 (39 uniq "( move ua , 3 ; move ua , 4 ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 1) (34 sub 3) (35 sub 3)
   (36 sub 1) (37 sub 3) (38 sub 3) (40 sub 3) (41 sub 1) (42 sub 2)
   (43 sub 2) (44 sub 2) (45 sub 3) (61 sub 2) (62 sub 2) (63 sub 1)
   (64 sub 1) (65 sub 3) (66 sub 3) (67 sub 3) (79 sub 3) (364 ins 4)
   (409 ins 4) (414 ins 4) (456 ins 5) (492 ins 5))
  ((1549 nul 13)) -1 0)
 (40 uniq "( move 4 , 2 ; move ua , ul ) "
  ((20 sub 2) (21 sub 3) (30 sub 3) (32 sub 3) (33 sub 3) (34 sub 3)
   (35 sub 3) (36 sub 3) (37 sub 2) (39 sub 3) (41 sub 3) (42 sub 3)
   (44 sub 2) (45 sub 3) (50 sub 3) (61 sub 3) (62 sub 1) (63 sub 3)
   (64 sub 3) (66 sub 3) (67 sub 3) (169 ins 2) (264 ins 3) (482 ins 5)
   (600 ins 5))
  ((120 nul 3) (712 nul 7) (836 nul 8) (1005 nul 8) (1199 nul 9)
   (1493 nul 12))
  -1 0)
 (41 uniq "( move ua , 4 ; move ua , 4 ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 1) (34 sub 3) (35 sub 3)
   (36 sub 2) (37 sub 3) (38 sub 3) (39 sub 1) (40 sub 3) (42 sub 3)
   (43 sub 2) (44 sub 2) (45 sub 2) (61 sub 2) (62 sub 2) (63 sub 1)
   (64 sub 1) (65 sub 3) (66 sub 3) (67 sub 3) (367 ins 4) (405 ins 4)
   (406 ins 4) (417 ins 4) (456 ins 5) (492 ins 5) (495 ins 5) (610 ins 5))
  ((122 nul 3) (406 ins 4) (495 ins 5) (765 nul 7) (780 nul 7)) -1 0)
 (42 uniq "( move ua , 3 ; move 4 , ul ) "
  ((21 sub 2) (32 sub 3) (33 sub 3) (35 sub 1) (36 sub 2) (38 sub 2)
   (39 sub 2) (40 sub 3) (41 sub 3) (43 sub 2) (45 sub 1) (54 sub 3)
   (55 sub 3) (60 sub 3) (61 sub 3) (62 sub 2) (63 sub 3) (64 sub 3)
   (65 sub 2) (66 sub 1) (67 sub 1) (89 sub 3) (382 ins 4) (505 ins 5)
   (531 ins 5))
  ((430 nul 6)) -1 0)
 (43 uniq "( move ua , ul ; move 4 , 4 ) "
  ((21 sub 1) (32 sub 3) (33 sub 2) (35 sub 2) (36 sub 3) (38 sub 1)
   (39 sub 2) (41 sub 2) (42 sub 2) (44 sub 2) (45 sub 2) (46 sub 3)
   (61 sub 3) (62 sub 3) (63 sub 2) (64 sub 1) (65 sub 1) (66 sub 2)
   (67 sub 1) (88 sub 3) (89 sub 3) (132 ins 2) (366 ins 4) (408 ins 4)
   (661 ins 6))
  ((170 nul 3) (360 nul 6) (655 nul 7) (743 nul 9) (824 nul 9) (1100 nul 11))
  -1 0)
 (44 uniq "( move 4 , ul ; move ua , 4 ) "
  ((20 sub 2) (21 sub 3) (32 sub 3) (33 sub 2) (34 sub 3) (36 sub 3)
   (37 sub 2) (38 sub 3) (39 sub 2) (40 sub 2) (41 sub 2) (43 sub 2)
   (48 sub 3) (61 sub 3) (62 sub 3) (63 sub 2) (64 sub 1) (65 sub 3)
   (67 sub 3) (80 sub 3) (131 ins 2) (174 ins 2) (177 ins 2) (332 ins 4)
   (334 ins 4) (336 ins 4) (373 ins 4) (380 ins 4) (455 ins 5) (507 ins 5)
   (617 ins 5))
  ((174 ins 2) (177 ins 2) (763 nul 8) (1300 nul 10) (1314 nul 10)) -1 0)
 (45 uniq "( move ua , 4 ; move 4 , ul ) "
  ((21 sub 2) (23 sub 3) (32 sub 3) (33 sub 3) (35 sub 1) (36 sub 3)
   (38 sub 2) (39 sub 3) (40 sub 3) (41 sub 2) (42 sub 1) (43 sub 2)
   (54 sub 3) (55 sub 2) (60 sub 2) (61 sub 3) (62 sub 2) (63 sub 3)
   (64 sub 3) (65 sub 2) (66 sub 1) (67 sub 1) (89 sub 3) (133 ins 2)
   (416 ins 4) (549 ins 5) (567 ins 5))
  ((119 nul 3) (656 nul 7) (746 nul 9) (769 nul 7) (1449 nul 12)) -1 0)
 (46 equiv "( pen ua + ul ; move 4 , 0 ) "
  ((12 del 2) (22 sub 1) (23 sub 1) (24 sub 2) (38 sub 3) (43 sub 3)
   (48 sub 3) (54 sub 3) (55 sub 2) (58 sub 2) (60 sub 3) (65 sub 3)
   (67 sub 3) (68 sub 2) (80 sub 3) (82 sub 3) (88 sub 2) (89 sub 1))
  () 12 0)
 (47 equiv "( pen ua ; move 2 + 2 , ul ) "
  ((19 sub 3) (69 sub 2) (70 sub 3) (77 sub 2) (78 sub 2) (560 ins 5)) () 16
  0)
 (48 equiv "( pen 2 + ul ; move ua , 4 ) "
  ((24 sub 2) (44 sub 3) (46 sub 3) (50 sub 3) (58 sub 3) (64 sub 3)
   (68 sub 3) (80 sub 2) (82 sub 2) (89 sub 3) (469 ins 5) (573 ins 5))
  () 13 0)
 (49 equiv "( pen 3 ; move ua , 3 + ul ) "
  ((25 sub 3) (26 sub 3) (31 sub 3) (52 sub 3) (57 sub 2) (81 sub 2)
   (83 sub 2) (95 sub 3) (516 ins 5) (532 ins 5))
  () 13 0)
 (50 equiv "( pen 2 * 2 ; move ua , ul ) "
  ((30 sub 2) (40 sub 3) (48 sub 3) (54 sub 2) (60 sub 3) (62 sub 3)
   (75 sub 2) (79 sub 3) (82 sub 3) (97 sub 3) (181 ins 2) (536 ins 5))
  () 17 0)
 (51 uniq "( pen ua ; move ua , 4 * 2 ) "
  ((25 sub 2) (26 sub 3) (70 sub 3) (73 sub 3) (74 sub 1) (76 sub 3)
   (77 sub 3) (87 sub 3) (95 sub 3))
  ((144 nul 3)) -1 0)
 (52 uniq "( pen 3 ; move ua , ua - 4 ) "
  ((13 del 2) (26 sub 3) (27 sub 3) (31 sub 3) (49 sub 3) (53 sub 2)
   (74 sub 3) (76 sub 3) (81 sub 3) (83 sub 1) (87 sub 2) (90 sub 3)
   (95 sub 3))
  ((268 nul 4)) -1 0)
 (53 uniq "( pen 2 ; move ua , ul - 4 ) "
  ((26 sub 2) (27 sub 3) (31 sub 3) (52 sub 2) (76 sub 2) (83 sub 3)
   (86 sub 3) (87 sub 1) (90 sub 2))
  () -1 0)
 (54 equiv "( pen ua * 2 ; move 4 , ul ) "
  ((16 del 2) (21 sub 3) (22 sub 2) (23 sub 3) (28 sub 3) (29 sub 3)
   (30 sub 1) (35 sub 3) (42 sub 3) (45 sub 3) (46 sub 3) (50 sub 2)
   (55 sub 2) (58 sub 3) (60 sub 1) (62 sub 3) (66 sub 3) (67 sub 3)
   (75 sub 2) (79 sub 3) (88 sub 3) (89 sub 2) (92 sub 3))
  () 16 0)
 (55 equiv "( pen ua + 4 ; move 4 , ul ) "
  ((16 del 2) (22 sub 2) (23 sub 1) (24 sub 3) (30 sub 3) (35 sub 3)
   (42 sub 3) (45 sub 2) (46 sub 2) (54 sub 2) (58 sub 2) (60 sub 1)
   (66 sub 3) (67 sub 3) (82 sub 3) (88 sub 3) (89 sub 1) (92 sub 3)
   (187 ins 2) (310 ins 3))
  () 16 0)
 (56 equiv "( pen 2 / ua ; move 4 , ua ) "
  ((15 del 2) (24 sub 3) (28 sub 3) (29 sub 3) (71 sub 3) (84 sub 3)
   (88 sub 3) (91 sub 3) (92 sub 2) (96 sub 2) (97 sub 3))
  () 0 0)
 (57 equiv "( pen 3 ; move 4 , ul + ul ) "
  ((49 sub 2) (59 sub 3) (81 sub 3) (83 sub 3)) () 11 0)
 (58 equiv "( pen 4 + ul ; move 4 , ul ) "
  ((22 sub 3) (23 sub 3) (24 sub 2) (29 sub 3) (30 sub 2) (46 sub 2)
   (48 sub 3) (54 sub 3) (55 sub 2) (60 sub 3) (67 sub 3) (80 sub 2)
   (82 sub 2) (88 sub 3) (89 sub 1) (92 sub 3))
  () 16 0)
 (59 uniq "( pen ua ; move 4 , ul / 3 ) "
  ((16 del 2) (27 sub 3) (57 sub 3) (76 sub 2) (87 sub 3)) () -1 0)
 (60 equiv "( pen ua * 4 ; move 4 , ul ) "
  ((16 del 2) (22 sub 3) (23 sub 2) (28 sub 3) (29 sub 3) (30 sub 2)
   (35 sub 3) (42 sub 3) (45 sub 2) (46 sub 3) (50 sub 3) (54 sub 1)
   (55 sub 1) (58 sub 3) (66 sub 3) (67 sub 3) (75 sub 3) (79 sub 3)
   (88 sub 3) (89 sub 2) (92 sub 3))
  () 16 0)
 (61 equiv "( move ua , ua ; move ua , 2 ) "
  ((20 sub 2) (21 sub 3) (32 sub 1) (33 sub 2) (34 sub 1) (35 sub 3)
   (36 sub 2) (37 sub 1) (38 sub 2) (39 sub 2) (40 sub 3) (41 sub 2)
   (42 sub 3) (43 sub 3) (44 sub 3) (45 sub 3) (62 sub 2) (63 sub 1)
   (64 sub 2) (65 sub 3) (66 sub 2) (67 sub 3) (84 sub 3) (149 ins 2)
   (191 ins 2) (203 ins 2) (274 ins 3) (314 ins 3) (364 ins 4) (393 ins 4)
   (394 ins 4) (399 ins 4) (402 ins 4) (429 ins 4) (433 ins 4) (435 ins 4)
   (438 ins 4) (439 ins 4) (446 ins 4) (534 ins 5) (537 ins 5) (582 ins 5)
   (584 ins 5) (635 ins 5))
  () 32 0)
 (62 uniq "( move ua , 2 ; move ua , ul ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 2) (34 sub 3) (35 sub 2)
   (36 sub 2) (37 sub 3) (38 sub 3) (39 sub 2) (40 sub 1) (41 sub 2)
   (42 sub 2) (43 sub 3) (44 sub 3) (45 sub 2) (50 sub 3) (54 sub 3)
   (61 sub 2) (63 sub 2) (64 sub 2) (65 sub 3) (66 sub 2) (67 sub 2)
   (75 sub 3) (160 ins 2) (164 ins 2) (192 ins 2) (195 ins 2) (292 ins 3)
   (342 ins 4) (356 ins 4) (395 ins 4) (448 ins 4) (517 ins 5) (562 ins 5)
   (579 ins 5) (599 ins 5) (738 ins 6))
  ((192 ins 2) (195 ins 2) (208 nul 3) (240 nul 3) (562 ins 5) (1513 nul 12))
  -1 0)
 (63 equiv "( move ua , ua ; move ua , 4 ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 1) (34 sub 2) (35 sub 3)
   (36 sub 2) (37 sub 2) (38 sub 3) (39 sub 1) (40 sub 3) (41 sub 1)
   (42 sub 3) (43 sub 2) (44 sub 2) (45 sub 3) (61 sub 1) (62 sub 2)
   (64 sub 1) (65 sub 3) (66 sub 2) (67 sub 3) (194 ins 2) (198 ins 2)
   (199 ins 2) (204 ins 2) (364 ins 4) (390 ins 4) (406 ins 4) (409 ins 4)
   (414 ins 4) (565 ins 5) (566 ins 5) (599 ins 5) (610 ins 5) (647 ins 5)
   (650 ins 5) (657 ins 5) (740 ins 6))
  () 33 0)
 (64 uniq "( move ua , ul ; move ua , 4 ) "
  ((20 sub 3) (21 sub 2) (32 sub 2) (33 sub 1) (34 sub 3) (35 sub 3)
   (36 sub 2) (37 sub 3) (38 sub 2) (39 sub 1) (40 sub 3) (41 sub 1)
   (42 sub 3) (43 sub 1) (44 sub 1) (45 sub 3) (48 sub 3) (61 sub 2)
   (62 sub 2) (63 sub 1) (65 sub 2) (66 sub 3) (67 sub 2) (68 sub 3)
   (161 ins 2) (163 ins 2) (168 ins 2) (178 ins 2) (197 ins 2) (198 ins 2)
   (204 ins 2) (205 ins 2) (366 ins 4) (407 ins 4) (447 ins 4) (508 ins 5)
   (565 ins 5) (605 ins 5) (610 ins 5) (611 ins 5) (650 ins 5) (657 ins 5)
   (740 ins 6))
  ((197 ins 2) (198 ins 2) (204 ins 2) (508 ins 5) (657 ins 5) (1104 nul 9)
   (1224 nul 10) (1397 nul 11))
  -1 0)
 (65 equiv "( move ua , ul ; move 4 , ua ) "
  ((21 sub 2) (32 sub 3) (33 sub 3) (35 sub 2) (36 sub 3) (38 sub 1)
   (39 sub 3) (41 sub 3) (42 sub 2) (43 sub 1) (44 sub 3) (45 sub 2)
   (46 sub 3) (61 sub 3) (62 sub 3) (63 sub 3) (64 sub 2) (66 sub 2)
   (67 sub 1) (88 sub 2) (89 sub 3) (555 ins 5) (672 ins 6))
  () 1 0)
 (66 equiv "( move ua , ua ; move 4 , ul ) "
  ((21 sub 2) (32 sub 3) (33 sub 3) (34 sub 3) (35 sub 1) (36 sub 3)
   (37 sub 3) (38 sub 2) (39 sub 3) (40 sub 3) (41 sub 3) (42 sub 1)
   (43 sub 2) (45 sub 1) (54 sub 3) (55 sub 3) (60 sub 3) (61 sub 2)
   (62 sub 2) (63 sub 2) (64 sub 3) (65 sub 2) (67 sub 1) (89 sub 3)
   (92 sub 3) (176 ins 2) (200 ins 2) (282 ins 3) (382 ins 4) (403 ins 4)
   (450 ins 4) (505 ins 5) (567 ins 5) (571 ins 5) (592 ins 5) (614 ins 5)
   (659 ins 5))
  () 35 0)
 (67 uniq "( move ua , ul ; move 4 , ul ) "
  ((21 sub 2) (32 sub 3) (33 sub 3) (35 sub 1) (36 sub 3) (38 sub 1)
   (39 sub 3) (40 sub 3) (41 sub 3) (42 sub 1) (43 sub 1) (44 sub 3)
   (45 sub 1) (46 sub 3) (54 sub 3) (55 sub 3) (58 sub 3) (60 sub 3)
   (61 sub 3) (62 sub 2) (63 sub 3) (64 sub 2) (65 sub 1) (66 sub 1)
   (88 sub 3) (89 sub 2) (330 ins 3) (415 ins 4) (416 ins 4) (450 ins 4)
   (500 ins 5) (506 ins 5) (549 ins 5) (556 ins 5) (559 ins 5) (563 ins 5)
   (567 ins 5) (574 ins 5))
  ((207 nul 3) (330 ins 3) (500 ins 5) (558 nul 6) (574 ins 5) (671 nul 7)
   (946 nul 8) (1253 nul 10) (1352 nul 13))
  -1 0)
 (68 equiv "( pen ua - ul ; move ua , 0 ) "
  ((14 del 2) (22 sub 3) (23 sub 3) (46 sub 2) (48 sub 3) (64 sub 3)
   (71 sub 3) (75 sub 3) (79 sub 3) (80 sub 3) (82 sub 3) (88 sub 2)
   (89 sub 3) (518 ins 5))
  () 14 0)
 (69 uniq "( pen ua ; move ua + 2 , ua ) "
  ((18 del 2) (19 sub 3) (47 sub 2) (70 sub 2) (73 sub 3) (77 sub 2)
   (78 sub 3) (86 sub 3) (95 sub 3) (179 ins 2) (182 ins 2) (519 ins 5))
  ((134 nul 4) (135 nul 5) (136 nul 5) (146 nul 4) (179 ins 2) (384 nul 5)
   (460 nul 6) (519 ins 5) (627 nul 6) (634 nul 6) (855 nul 8) (914 nul 11)
   (972 nul 9) (1019 nul 15) (1030 nul 9) (1041 nul 13) (1079 nul 12)
   (1254 nul 12) (1404 nul 14) (1523 nul 15) (1536 nul 14))
  -1 0)
 (70 uniq "( pen ua ; move ua + ul , 2 ) "
  ((47 sub 3) (51 sub 3) (69 sub 2) (72 sub 3) (73 sub 1) (74 sub 3)
   (76 sub 3) (77 sub 2) (78 sub 2) (86 sub 2) (87 sub 3) (215 ins 2)
   (216 ins 2) (640 ins 5))
  ((138 nul 3) (215 ins 2) (216 ins 2) (572 nul 7) (586 nul 9) (640 ins 5)
   (1046 nul 10) (1265 nul 10) (1540 nul 13))
  -1 0)
 (71 equiv "( pen ul / ua ; move ua , 0 ) "
  ((14 del 2) (56 sub 3) (68 sub 3) (84 sub 1) (91 sub 1) (92 sub 2)
   (96 sub 1) (97 sub 1) (338 ins 3) (602 ins 5) (609 ins 5) (664 ins 5)
   (708 ins 6))
  () 0 0)
 (72 uniq "( pen ua ; move ul - ua , 2 ) "
  ((70 sub 3) (73 sub 2) (74 sub 3) (85 sub 1) (86 sub 3) (569 ins 5)
   (603 ins 5) (625 ins 5))
  ((73 sub 2) (509 nul 7) (569 ins 5) (603 ins 5) (625 ins 5) (917 nul 9)
   (1538 nul 13))
  -1 0)
 (73 equiv "( pen ua ; move ua - ul , 2 ) "
  ((51 sub 3) (69 sub 3) (70 sub 1) (72 sub 2) (74 sub 3) (76 sub 3)
   (77 sub 3) (78 sub 3) (85 sub 3) (86 sub 1) (87 sub 3))
  () 72 0)
 (74 equiv "( pen ua ; move ua , ua * 2 ) "
  ((18 del 2) (25 sub 3) (51 sub 1) (52 sub 3) (70 sub 3) (72 sub 3)
   (73 sub 3) (76 sub 3) (81 sub 3) (83 sub 3) (87 sub 3) (95 sub 2)
   (221 ins 2) (625 ins 5))
  () 14 0)
 (75 equiv "( pen ua * 2 ; move ua , ua ) "
  ((18 del 2) (22 sub 3) (28 sub 3) (29 sub 3) (30 sub 3) (50 sub 2)
   (54 sub 2) (60 sub 3) (62 sub 3) (68 sub 3) (79 sub 1) (80 sub 3)
   (88 sub 3) (96 sub 3) (179 ins 2))
  () 14 0)
 (76 uniq "( pen ua ; move ua , ul - 3 ) "
  ((27 sub 3) (51 sub 3) (52 sub 3) (53 sub 2) (59 sub 2) (70 sub 3)
   (73 sub 3) (74 sub 3) (83 sub 3) (86 sub 3) (87 sub 1) (90 sub 2)
   (95 sub 3) (229 ins 2) (307 ins 3) (649 ins 5))
  ((307 ins 3) (938 nul 9) (939 nul 10)) -1 0)
 (77 uniq "( pen ua ; move ua + 4 , ul ) "
  ((16 del 2) (47 sub 2) (51 sub 3) (69 sub 2) (70 sub 2) (73 sub 3)
   (78 sub 2) (86 sub 3) (217 ins 2) (339 ins 3))
  ((142 nul 3) (217 ins 2) (339 ins 3) (773 nul 12) (783 nul 7) (965 nul 7)
   (1530 nul 15))
  -1 0)
 (78 equiv "( pen ua ; move 3 + ul , ul ) "
  ((47 sub 2) (69 sub 3) (70 sub 2) (73 sub 3) (77 sub 2) (86 sub 3)
   (230 ins 2) (425 ins 4) (458 ins 4) (545 ins 5) (645 ins 5))
  () 16 0)
 (79 equiv "( pen ua * 3 ; move ua , ua ) "
  ((18 del 2) (28 sub 2) (29 sub 2) (36 sub 3) (39 sub 3) (50 sub 3)
   (54 sub 3) (60 sub 3) (68 sub 3) (75 sub 1) (80 sub 3) (88 sub 3)
   (96 sub 3) (611 ins 5))
  () 14 0)
 (80 equiv "( pen 4 + ul ; move ua , ua ) "
  ((24 sub 3) (29 sub 3) (44 sub 3) (46 sub 3) (48 sub 2) (58 sub 2)
   (68 sub 3) (75 sub 3) (79 sub 3) (82 sub 2) (88 sub 3) (89 sub 3)
   (96 sub 3) (384 ins 4) (542 ins 5) (627 ins 5))
  () 14 0)
 (81 equiv "( pen 4 ; move ua , ua + ul ) "
  ((17 del 2) (25 sub 2) (26 sub 3) (31 sub 3) (49 sub 2) (52 sub 3)
   (57 sub 3) (74 sub 3) (83 sub 2) (90 sub 3) (95 sub 2))
  () 17 0)
 (82 equiv "( pen 3 + ul ; move ua , ul ) "
  ((24 sub 3) (46 sub 3) (48 sub 2) (50 sub 3) (55 sub 3) (58 sub 2)
   (68 sub 3) (80 sub 2) (89 sub 2) (97 sub 3) (219 ins 2) (306 ins 3)
   (546 ins 5) (593 ins 5))
  () 17 0)
 (83 uniq "( pen 3 ; move ua , ua - ul ) "
  ((49 sub 2) (52 sub 1) (53 sub 3) (57 sub 3) (74 sub 3) (76 sub 3)
   (81 sub 2) (87 sub 3) (90 sub 3) (95 sub 3) (576 ins 5))
  ((143 nul 5) (576 ins 5) (1196 nul 11) (1426 nul 14)) -1 0)
 (84 equiv "( pen ul / ua ; move ua , 2 ) "
  ((34 sub 3) (37 sub 3) (56 sub 3) (61 sub 3) (71 sub 1) (91 sub 1)
   (92 sub 2) (96 sub 1) (97 sub 1) (222 ins 2))
  () 0 0)
 (85 uniq "( pen ua ; move ul - ua , 4 ) "
  ((72 sub 1) (73 sub 3) (86 sub 2) (667 ins 5))
  ((86 sub 2) (147 nul 3) (665 nul 7) (667 ins 5) (1190 nul 9) (1441 nul 13))
  -1 0)
 (86 equiv "( pen ua ; move ua - ul , 4 ) "
  ((53 sub 3) (69 sub 3) (70 sub 2) (72 sub 3) (73 sub 1) (76 sub 3)
   (77 sub 3) (78 sub 3) (85 sub 2) (87 sub 2) (617 ins 5) (650 ins 5)
   (665 ins 5))
  () 85 0)
 (87 uniq "( pen ua ; move ua , ul - 4 ) "
  ((26 sub 3) (27 sub 2) (31 sub 3) (51 sub 3) (52 sub 2) (53 sub 1)
   (59 sub 3) (70 sub 3) (73 sub 3) (74 sub 3) (76 sub 1) (83 sub 3)
   (86 sub 2) (90 sub 2) (95 sub 3))
  ((457 nul 7) (1528 nul 14)) -1 0)
 (88 equiv "( pen ua - ul ; move 4 , ua ) "
  ((15 del 2) (22 sub 3) (23 sub 3) (24 sub 3) (28 sub 3) (29 sub 3)
   (38 sub 3) (43 sub 3) (46 sub 2) (54 sub 3) (55 sub 3) (56 sub 3)
   (58 sub 3) (60 sub 3) (65 sub 2) (67 sub 3) (68 sub 2) (75 sub 3)
   (79 sub 3) (80 sub 3) (89 sub 2) (633 ins 5))
  () 12 0)
 (89 equiv "( pen ua + ul ; move 4 , ul ) "
  ((16 del 2) (22 sub 2) (23 sub 2) (24 sub 2) (30 sub 3) (35 sub 3)
   (38 sub 3) (42 sub 3) (43 sub 3) (45 sub 3) (46 sub 1) (48 sub 3)
   (54 sub 2) (55 sub 1) (58 sub 1) (60 sub 2) (65 sub 3) (66 sub 3)
   (67 sub 2) (68 sub 3) (80 sub 3) (82 sub 2) (88 sub 2) (92 sub 3)
   (308 ins 3))
  () 16 0)
 (90 equiv "( pen 4 ; move ua , ul - ua ) "
  ((17 del 2) (25 sub 3) (31 sub 3) (52 sub 3) (53 sub 2) (76 sub 2)
   (81 sub 3) (83 sub 3) (87 sub 2) (95 sub 3))
  () 17 0)
 (91 equiv "( pen ul / ua ; move ua , 3 ) "
  ((56 sub 3) (71 sub 1) (84 sub 1) (92 sub 2) (96 sub 1) (97 sub 1)
   (228 ins 2) (229 ins 2) (340 ins 3))
  () 0 0)
 (92 equiv "( pen ul / ua ; move 4 , ul ) "
  ((16 del 2) (30 sub 3) (54 sub 3) (55 sub 3) (56 sub 2) (58 sub 3)
   (60 sub 3) (66 sub 3) (71 sub 2) (84 sub 2) (89 sub 3) (91 sub 2)
   (96 sub 2) (97 sub 1))
  () 0 0)
 (93 equiv "( pen 2 ; pen 3 ; move 4 , 2 ) "
  ((11 del 3) (94 sub 1) (99 sub 3) (110 sub 3)) () 11 0)
 (94 equiv "( pen 3 ; pen 3 ; move 4 , 2 ) "
  ((11 del 3) (93 sub 1) (99 sub 3) (110 sub 3)) () 11 0)
 (95 equiv "( pen ua ; move ua , ua + ua ) "
  ((18 del 2) (25 sub 3) (26 sub 3) (49 sub 3) (51 sub 3) (52 sub 3)
   (69 sub 3) (74 sub 2) (76 sub 3) (81 sub 2) (83 sub 3) (87 sub 3)
   (90 sub 3) (642 ins 5))
  () 14 0)
 (96 equiv "( pen ul / ua ; move ua , ua ) "
  ((18 del 2) (56 sub 2) (71 sub 1) (75 sub 3) (79 sub 3) (80 sub 3)
   (84 sub 1) (91 sub 1) (92 sub 2) (97 sub 1) (222 ins 2) (225 ins 2)
   (247 ins 2) (385 ins 3))
  () 0 0)
 (97 equiv "( pen ul / ua ; move ua , ul ) "
  ((50 sub 3) (56 sub 3) (71 sub 1) (82 sub 3) (84 sub 1) (91 sub 1)
   (92 sub 1) (96 sub 1) (229 ins 2) (231 ins 2) (248 ins 2) (340 ins 3)
   (609 ins 5) (664 ins 5))
  () 0 0)
 (98 equiv "( pen ua ; move ua , 0 ; pen 2 ) "
  ((14 del 3) (100 sub 2) (111 sub 3)) () 14 0)
 (99 equiv "( pen ua ; pen ua ; move 4 , 0 ) "
  ((12 del 3) (93 sub 3) (94 sub 3) (110 sub 1)) () 12 0)
 (100 equiv "( pen ua ; move 4 , 0 ; pen ul ) "
  ((12 del 3) (16 del 3) (98 sub 2) (111 sub 1)) () 12 0)
 (101 uniq "( move 4 , 0 ; move ua + 2 , 2 ) "
  ((20 del 2) (113 sub 3) (115 sub 1) (116 sub 3) (153 sub 3) (161 sub 3)
   (166 sub 2) (193 sub 3))
  () -1 0)
 (102 uniq "( move ua , 0 ; move 4 , 3 + 2 ) "
  ((113 sub 2) (148 sub 2) (155 sub 2) (159 sub 3)) () -1 0)
 (103 uniq "( move ua , 3 + 2 ; move 4 , 2 ) "
  ((107 sub 3) (119 sub 2) (128 sub 3) (129 sub 3) (149 sub 2) (160 sub 2)
   (167 sub 3) (176 sub 3) (192 sub 3) (200 sub 3))
  () -1 0)
 (104 uniq "( move ua , 3 ; move 3 * 2 , 2 ) " ((162 sub 3)) () -1 0)
 (105 uniq "( move 3 * 3 , 2 ; move 2 , ul ) " ((120 sub 3)) () -1 0)
 (106 uniq "( move ua , 2 ; move 4 + 4 , 4 ) "
  ((21 del 2) (118 sub 2) (161 sub 3) (168 sub 3) (170 sub 3)) () -1 0)
 (107 equiv "( move ua , 4 - 2 ; move 4 , 4 ) "
  ((21 del 2) (103 sub 3) (119 sub 3) (128 sub 2) (129 sub 3) (132 sub 2)
   (167 sub 3) (171 sub 3) (176 sub 3) (204 sub 3))
  () 21 0)
 (108 uniq "( move 2 , ul ; move 3 * 3 , 4 ) " ((168 sub 3) (170 sub 3)) ()
  -1 0)
 (109 equiv "( pen ua - 3 ; move 0 - 4 , 2 ) " ((11 del 4)) () 11 0)
 (110 equiv "( pen ua ; pen ua ; move 4 , ua ) "
  ((15 del 3) (93 sub 3) (94 sub 3) (99 sub 1)) () 12 0)
 (111 equiv "( pen ua ; move 4 , ua ; pen ul ) "
  ((15 del 3) (16 del 3) (98 sub 3) (100 sub 1) (578 ins 4)) () 12 0)
 (112 uniq "( move ua , 0 ; move 0 - ua , 2 ) "
  ((32 del 2) (148 sub 3) (152 sub 3) (241 sub 3)) () -1 0)
 (113 equiv "( move ua , 0 ; move ua , 2 + 2 ) "
  ((32 del 2) (101 sub 3) (102 sub 2) (114 sub 3) (116 sub 3) (122 sub 2)
   (123 sub 3) (148 sub 1) (150 sub 3) (155 sub 3) (165 sub 3) (194 sub 3)
   (197 sub 3) (236 sub 3))
  () 33 0)
 (114 uniq "( move 3 , 0 ; move ua , ua - 2 ) "
  ((113 sub 3) (116 sub 3) (148 sub 2) (150 sub 2) (165 sub 3) (173 sub 3)
   (205 sub 3) (213 sub 3))
  ((392 nul 4) (520 nul 5)) -1 0)
 (115 uniq "( move 4 , 0 ; move ua + 2 , ul ) "
  ((101 sub 1) (116 sub 3) (152 sub 3) (153 sub 3) (154 sub 3) (161 sub 3)
   (166 sub 1) (212 sub 3))
  ((166 sub 1) (1270 nul 9)) -1 0)
 (116 equiv "( move 4 , 0 ; move ua , 2 - ua ) "
  ((20 del 2) (101 sub 3) (113 sub 3) (114 sub 3) (115 sub 3) (131 sub 3)
   (150 sub 3) (173 sub 3) (177 sub 2) (199 sub 3) (213 sub 3))
  () 20 0)
 (117 equiv "( move 2 * 2 , ua ; move ua , 2 ) "
  ((120 sub 2) (124 sub 1) (127 sub 2) (158 sub 3) (191 sub 3)) () 20 0)
 (118 uniq "( move ua , 2 ; move ua + 3 , 4 ) "
  ((106 sub 2) (152 sub 3) (161 sub 2) (168 sub 1) (170 sub 3) (193 sub 3))
  () -1 0)
 (119 equiv "( move ua , 2 + 2 ; move 4 , ul ) "
  ((103 sub 2) (107 sub 3) (133 sub 2) (149 sub 3) (160 sub 2) (169 sub 3)
   (176 sub 2) (192 sub 2) (195 sub 3) (200 sub 2) (240 sub 3))
  () 45 0)
 (120 equiv "( move 2 * 2 , 2 ; move ua , ul ) "
  ((105 sub 3) (117 sub 2) (124 sub 3) (678 ins 4)) () 40 0)
 (121 equiv "( move ua + 4 , ul ; v0 = 0 ) " ((151 sub 3) (783 ins 5)) () 1
  0)
 (122 equiv "( move ua , 4 ; move ua , 2 * 2 ) "
  ((113 sub 2) (148 sub 3) (178 sub 3) (235 sub 3) (242 sub 3)) () 41 0)
 (123 uniq "( move ua , 0 ; move ua , 4 / 3 ) "
  ((33 del 2) (113 sub 3) (148 sub 3) (150 sub 3) (164 sub 3) (165 sub 2)
   (178 sub 3) (199 sub 3))
  () -1 0)
 (124 uniq "( move 4 * 2 , ua ; move ua , 2 ) "
  ((37 del 2) (117 sub 1) (120 sub 3) (125 sub 3) (127 sub 1) (158 sub 3)
   (191 sub 3))
  ((705 nul 8)) -1 0)
 (125 uniq "( move 4 + ul , 4 ; move ua , 2 ) "
  ((124 sub 3) (157 sub 3) (158 sub 2) (172 sub 3) (191 sub 2) (196 sub 3)
   (202 sub 3) (812 ins 5))
  ((412 nul 3) (1107 nul 8)) -1 0)
 (126 uniq "( move ua , 2 ; move 4 , 2 / ua ) " ((159 sub 3)) () -1 0)
 (127 uniq "( move 4 * 2 , ua ; move ul , 2 ) " ((117 sub 2) (124 sub 1)) ()
  -1 0)
 (128 uniq "( move ua , 3 - ua ; move 4 , 4 ) "
  ((103 sub 3) (107 sub 2) (129 sub 2) (132 sub 2) (167 sub 2) (175 sub 3)
   (195 sub 3) (200 sub 3) (204 sub 2) (208 sub 3))
  () -1 0)
 (129 equiv "( move ua , 3 - 4 ; move 4 , ua ) "
  ((103 sub 3) (107 sub 3) (128 sub 2) (132 sub 2) (133 sub 3) (167 sub 3)
   (208 sub 3))
  () 2 0)
 (130 equiv "( move 4 , 4 - 3 ; move ua , ua ) "
  ((156 sub 3) (171 sub 3) (175 sub 3) (210 sub 2) (211 sub 2)) () 1 0)
 (131 uniq "( move 4 , ul ; move ua , 3 - 4 ) "
  ((44 del 2) (116 sub 3) (150 sub 3) (168 sub 3) (173 sub 2) (174 sub 2)
   (177 sub 2) (178 sub 3) (197 sub 3) (205 sub 2) (213 sub 2))
  ((173 sub 2) (176 nul 8) (213 sub 2)) -1 0)
 (132 uniq "( move ua , ul - 4 ; move 4 , 4 ) "
  ((43 del 2) (107 sub 2) (128 sub 2) (129 sub 2) (133 sub 3) (167 sub 2)
   (204 sub 2) (761 ins 5))
  ((1321 nul 11)) -1 0)
 (133 uniq "( move ua , 2 / 4 ; move 4 , ul ) "
  ((45 del 2) (119 sub 2) (129 sub 3) (132 sub 3) (176 sub 2) (195 sub 3)
   (200 sub 3) (243 sub 3))
  () -1 0)
 (134 equiv "( pen 4 + ul ; move ua + 2 , 0 ) " ((138 sub 3)) () 69 0)
 (135 equiv "( pen ua + ua ; move 4 * 2 , 0 ) " ((12 del 4) (136 sub 2)) ()
  69 0)
 (136 equiv "( pen ua + ua ; move 4 + 4 , 0 ) "
  ((12 del 4) (135 sub 2) (217 sub 2)) () 69 0)
 (137 uniq "( pen ua ; move ua + 3 , 2 - 4 ) "
  ((183 sub 3) (216 sub 2) (218 sub 2)) ((1012 nul 8)) -1 0)
 (138 equiv "( pen 4 + 3 ; move ua + ul , 2 ) " ((134 sub 3)) () 70 0)
 (139 equiv "( pen 3 + 2 ; move 3 + ul , ul ) " ((142 sub 3) (230 sub 3)) ()
  16 0)
 (140 uniq "( pen ua - 3 ; move 3 * 3 , ua ) " ((186 sub 3)) () -1 0)
 (141 uniq "( pen 3 ; move ua + 2 , ua / 4 ) "
  ((13 del 4) (182 sub 3) (184 sub 3)) () -1 0)
 (142 equiv "( pen 3 * 2 ; move ua + 4 , ul ) "
  ((139 sub 3) (179 sub 3) (217 sub 3)) () 77 0)
 (143 equiv "( pen ua / 2 ; move ua , 3 - 4 ) " () () 83 0)
 (144 equiv "( pen 4 + ul ; move ua , 4 * 2 ) " ((221 sub 3)) () 51 0)
 (145 uniq "( pen 4 ; move 4 + ul , 3 - ua ) " ((216 sub 3))
  ((557 nul 12) (764 nul 8) (923 nul 8)) -1 0)
 (146 equiv "( pen ua ; move 4 * 2 , ua * 4 ) " ((15 del 4)) () 69 0)
 (147 equiv "( pen 3 * 3 ; move ul - ua , 4 ) " ((13 del 4)) () 85 0)
 (148 equiv "( move ua , 0 ; move ua , ua + 2 ) "
  ((32 del 2) (102 sub 2) (112 sub 3) (113 sub 1) (114 sub 2) (122 sub 3)
   (123 sub 3) (150 sub 2) (155 sub 2) (159 sub 3) (165 sub 2) (174 sub 3)
   (194 sub 2) (197 sub 2) (205 sub 3) (235 sub 3) (236 sub 3) (242 sub 3)
   (741 ins 5))
  () 32 0)
 (149 uniq "( move ua , ua + 2 ; move ua , 2 ) "
  ((61 del 2) (103 sub 2) (119 sub 3) (160 sub 2) (176 sub 3) (192 sub 1)
   (198 sub 2) (200 sub 3) (203 sub 3) (240 sub 3))
  ((278 nul 5) (437 nul 6) (1304 nul 10)) -1 0)
 (150 uniq "( move ua , 0 ; move ua , ua - 4 ) "
  ((33 del 2) (113 sub 3) (114 sub 2) (116 sub 3) (123 sub 3) (131 sub 3)
   (148 sub 2) (155 sub 3) (159 sub 3) (164 sub 3) (165 sub 1) (174 sub 3)
   (178 sub 3) (194 sub 2) (197 sub 2) (199 sub 3) (205 sub 1) (213 sub 3)
   (235 sub 3) (242 sub 3))
  () -1 0)
 (151 equiv "( move ua + 4 , ul ; move 0 , ua ) "
  ((121 sub 3) (163 sub 3) (202 sub 3) (206 sub 3) (209 sub 3) (223 sub 3)
   (391 ins 2) (762 ins 5) (823 ins 5))
  () 1 0)
 (152 uniq "( move ua , 0 ; move ua - 3 , ul ) "
  ((112 sub 3) (115 sub 3) (118 sub 3) (154 sub 3) (168 sub 3) (212 sub 3)
   (241 sub 3))
  ((396 nul 4) (402 nul 5) (1021 nul 7)) -1 0)
 (153 equiv "( move 2 , ul ; move ua + 2 , ua ) "
  ((101 sub 3) (115 sub 3) (161 sub 2) (166 sub 3) (168 sub 3) (193 sub 3))
  () 1 0)
 (154 equiv "( move ua , 0 ; move 3 + ul , ul ) "
  ((115 sub 3) (152 sub 3) (170 sub 2) (193 sub 3) (201 sub 1) (207 sub 1))
  () 35 0)
 (155 equiv "( move ua , 0 ; move 4 , ua + ul ) "
  ((35 del 2) (102 sub 2) (113 sub 3) (148 sub 2) (150 sub 3) (159 sub 2)
   (165 sub 3) (187 sub 3) (194 sub 3) (197 sub 3) (236 sub 3) (742 ins 5))
  () 35 0)
 (156 uniq "( move ua , 0 - 3 ; move ua , ul ) "
  ((130 sub 3) (160 sub 3) (171 sub 3) (175 sub 3) (192 sub 3) (195 sub 2)
   (204 sub 3) (208 sub 2) (240 sub 3) (243 sub 3))
  ((1130 nul 9)) -1 0)
 (157 uniq "( move ua + ul , 2 ; move ua , 3 ) "
  ((125 sub 3) (158 sub 3) (163 sub 3) (172 sub 3) (191 sub 2) (196 sub 3)
   (202 sub 3) (206 sub 3) (239 sub 3))
  ((440 nul 3) (1557 nul 11)) -1 0)
 (158 equiv "( move 3 + ul , ua ; move ua , 2 ) "
  ((34 del 2) (117 sub 3) (124 sub 3) (125 sub 2) (157 sub 3) (191 sub 1)
   (196 sub 3) (202 sub 2) (203 sub 3) (239 sub 3) (679 ins 4))
  () 20 0)
 (159 uniq "( move ua , 0 ; move 4 , ua * ua ) "
  ((102 sub 3) (126 sub 3) (148 sub 3) (150 sub 3) (155 sub 2) (165 sub 3)
   (235 sub 2) (242 sub 2))
  ((910 nul 6)) -1 0)
 (160 uniq "( move ua , 3 + 2 ; move ua , ul ) "
  ((62 del 2) (103 sub 2) (119 sub 2) (149 sub 2) (156 sub 3) (169 sub 3)
   (175 sub 3) (176 sub 3) (192 sub 1) (195 sub 3) (198 sub 3) (200 sub 3)
   (208 sub 2) (240 sub 2) (243 sub 3) (342 ins 2))
  () -1 0)
 (161 uniq "( move ua , ul ; move ua + 2 , 4 ) "
  ((64 del 2) (101 sub 3) (106 sub 3) (115 sub 3) (118 sub 2) (153 sub 2)
   (166 sub 3) (168 sub 1) (170 sub 2) (178 sub 3) (193 sub 2) (197 sub 3)
   (205 sub 3) (207 sub 3))
  ((1518 nul 10)) -1 0)
 (162 equiv "( move ua , ua ; move 2 * 2 , ul ) "
  ((104 sub 3) (166 sub 3) (201 sub 3) (241 sub 3)) () 35 0)
 (163 uniq "( move ua + 2 , ul ; move ua , 4 ) "
  ((64 del 2) (151 sub 3) (157 sub 3) (191 sub 3) (196 sub 3) (198 sub 3)
   (202 sub 3) (206 sub 3) (209 sub 3) (366 ins 2))
  ((1491 nul 10)) -1 0)
 (164 uniq "( move ua , 2 ; move ua , ul - 3 ) "
  ((62 del 2) (123 sub 3) (150 sub 3) (173 sub 3) (199 sub 3) (205 sub 3)
   (236 sub 3) (356 ins 2) (395 ins 2))
  ((394 nul 5) (395 ins 2) (448 nul 4)) -1 0)
 (165 uniq "( move ua , 0 ; move ua , ua / 4 ) "
  ((33 del 2) (113 sub 3) (114 sub 3) (123 sub 2) (148 sub 2) (150 sub 1)
   (155 sub 3) (159 sub 3) (174 sub 3) (178 sub 3) (194 sub 2) (197 sub 2)
   (205 sub 2) (235 sub 3) (242 sub 3))
  () -1 0)
 (166 equiv "( move 4 , ua ; move ua + 2 , ul ) "
  ((101 sub 2) (115 sub 1) (153 sub 3) (161 sub 3) (162 sub 3) (201 sub 3)
   (212 sub 2))
  () 115 0)
 (167 equiv "( move ua , ul - ua ; move 4 , 2 ) "
  ((38 del 2) (103 sub 3) (107 sub 3) (128 sub 2) (129 sub 3) (132 sub 2)
   (191 sub 3) (195 sub 3) (200 sub 3) (203 sub 2) (204 sub 2) (210 sub 3)
   (243 sub 3) (915 ins 5))
  () 38 0)
 (168 uniq "( move ua , ul ; move ua + 3 , 4 ) "
  ((64 del 2) (106 sub 3) (108 sub 3) (118 sub 1) (131 sub 3) (152 sub 3)
   (153 sub 3) (161 sub 1) (170 sub 2) (178 sub 3) (193 sub 2) (197 sub 3)
   (205 sub 3) (207 sub 3))
  () -1 0)
 (169 uniq "( move 4 , 2 + ul ; move ua , ul ) "
  ((40 del 2) (119 sub 3) (160 sub 3) (175 sub 3) (192 sub 3) (195 sub 3)
   (198 sub 3) (208 sub 3) (211 sub 3) (240 sub 2))
  ((175 sub 3) (497 nul 6) (553 nul 8)) -1 0)
 (170 equiv "( move ua , ul ; move 3 + ul , 4 ) "
  ((106 sub 3) (108 sub 3) (118 sub 3) (154 sub 2) (161 sub 2) (168 sub 2)
   (193 sub 2) (201 sub 2) (207 sub 1))
  () 43 0)
 (171 equiv "( move ua , 4 - ul ; move ua , 3 ) "
  ((107 sub 3) (130 sub 3) (156 sub 3) (195 sub 3) (198 sub 3) (204 sub 3)
   (208 sub 2) (211 sub 3) (240 sub 3))
  () 36 0)
 (172 uniq "( move ua + ul , 4 ; move 3 , ul ) "
  ((125 sub 3) (157 sub 3) (191 sub 3) (206 sub 1) (209 sub 3) (239 sub 3)
   (377 ins 2))
  () -1 0)
 (173 equiv "( move 4 , ul ; move ua , ul - 2 ) "
  ((114 sub 3) (116 sub 3) (131 sub 2) (164 sub 3) (174 sub 3) (177 sub 2)
   (193 sub 3) (205 sub 3) (213 sub 2))
  () 131 0)
 (174 equiv "( move 4 , ul ; move ua , ua + 4 ) "
  ((44 del 2) (131 sub 2) (148 sub 3) (150 sub 3) (165 sub 3) (173 sub 3)
   (177 sub 3) (178 sub 3) (194 sub 2) (197 sub 1) (205 sub 2) (213 sub 2)
   (242 sub 3))
  () 44 0)
 (175 equiv "( move 4 , 3 - ua ; move ua , ul ) "
  ((128 sub 3) (130 sub 3) (156 sub 3) (160 sub 3) (169 sub 3) (195 sub 2)
   (204 sub 3) (208 sub 2) (210 sub 2) (211 sub 3) (243 sub 3))
  () 169 0)
 (176 equiv "( move ua , ua / 2 ; move 4 , ul ) "
  ((66 del 2) (103 sub 3) (107 sub 3) (119 sub 2) (133 sub 2) (149 sub 3)
   (160 sub 3) (192 sub 2) (200 sub 2) (243 sub 3))
  () 131 0)
 (177 equiv "( move 4 , ul ; move ua , 4 - ua ) "
  ((44 del 2) (116 sub 2) (131 sub 2) (173 sub 2) (174 sub 3) (178 sub 3)
   (199 sub 2) (205 sub 3) (213 sub 2) (242 sub 3))
  () 44 0)
 (178 uniq "( move ua , ul ; move ua , 4 * 4 ) "
  ((64 del 2) (122 sub 3) (123 sub 3) (131 sub 3) (150 sub 3) (161 sub 3)
   (165 sub 3) (168 sub 3) (174 sub 3) (177 sub 3) (194 sub 3) (197 sub 2)
   (199 sub 3) (205 sub 2) (235 sub 3) (242 sub 2))
  () -1 0)
 (179 equiv "( pen ua * 2 ; move ua + 2 , ua ) "
  ((18 del 4) (69 del 2) (75 del 2) (142 sub 3) (186 sub 3)) () 69 0)
 (180 equiv "( pen 2 + ul ; move ua , 3 + ul ) " ((181 sub 3) (219 sub 2)) ()
  13 0)
 (181 equiv "( pen 2 * 2 ; move ua , ua + ul ) "
  ((50 del 2) (180 sub 3) (219 sub 3)) () 17 0)
 (182 uniq "( pen ua ; move ua + 2 , 3 / ua ) "
  ((18 del 4) (69 del 2) (141 sub 3) (216 sub 3)) () -1 0)
 (183 uniq "( pen ua ; move 3 + 3 , ua - ul ) "
  ((137 sub 3) (184 sub 3) (218 sub 1)) () -1 0)
 (184 uniq "( pen 4 ; move ua + 2 , ua - ul ) "
  ((17 del 4) (141 sub 3) (183 sub 3) (218 sub 2)) () -1 0)
 (185 equiv "( pen ul ; move ua + 4 , ul - 2 ) " () () 2 0)
 (186 equiv "( pen ua * 3 ; move 3 + ul , ua ) " ((140 sub 3) (179 sub 3)) ()
  12 0)
 (187 equiv "( pen ua + 4 ; move 4 , ua + ul ) "
  ((15 del 4) (16 del 4) (55 del 2) (155 sub 3) (219 sub 3) (943 ins 5)) ()
  16 0)
 (188 equiv "( pen ul / ua ; move 4 + 2 , ua ) "
  ((15 del 4) (222 sub 3) (230 sub 3)) () 0 0)
 (189 uniq "( pen 2 ; move ul - ua , ua / 4 ) " () () -1 0)
 (190 equiv "( pen ul / ua ; move 3 * 3 , ul ) " ((230 sub 2) (248 sub 3)) ()
  0 0)
 (191 uniq "( move ua + ul , ua ; move ua , 2 ) "
  ((61 del 2) (117 sub 3) (124 sub 3) (125 sub 2) (157 sub 2) (158 sub 1)
   (163 sub 3) (167 sub 3) (172 sub 3) (196 sub 2) (202 sub 3) (203 sub 2)
   (204 sub 3) (206 sub 3) (239 sub 2) (243 sub 3) (393 ins 2) (439 ins 2)
   (446 ins 2) (582 ins 3) (899 ins 5))
  ((441 nul 3)) -1 0)
 (192 equiv "( move ua , ua + 2 ; move ua , ul ) "
  ((62 del 2) (103 sub 3) (119 sub 2) (149 sub 1) (156 sub 3) (160 sub 1)
   (169 sub 3) (176 sub 2) (195 sub 3) (198 sub 2) (200 sub 2) (208 sub 3)
   (240 sub 2) (243 sub 3) (395 ins 2))
  () 62 0)
 (193 uniq "( move ua , ul ; move ua + ul , 2 ) "
  ((101 sub 3) (118 sub 3) (153 sub 3) (154 sub 3) (161 sub 2) (168 sub 2)
   (170 sub 2) (173 sub 3) (201 sub 3) (207 sub 2) (434 ins 2) (435 ins 2)
   (942 ins 5) (1050 ins 6))
  ((434 ins 2) (435 ins 2) (487 nul 3) (626 nul 7) (804 nul 5) (901 nul 7))
  -1 0)
 (194 equiv "( move ua , ua ; move ua , ua + 4 ) "
  ((63 del 2) (113 sub 3) (148 sub 2) (150 sub 2) (155 sub 3) (165 sub 2)
   (174 sub 2) (178 sub 3) (197 sub 1) (199 sub 3) (205 sub 2) (225 sub 3)
   (235 sub 2) (236 sub 2) (242 sub 3) (740 ins 4))
  () 33 0)
 (195 equiv "( move ua , 2 - ua ; move ua , ul ) "
  ((62 del 2) (119 sub 3) (128 sub 3) (133 sub 3) (156 sub 2) (160 sub 3)
   (167 sub 3) (169 sub 3) (171 sub 3) (175 sub 2) (192 sub 3) (200 sub 3)
   (203 sub 3) (204 sub 2) (208 sub 2) (210 sub 3) (239 sub 3) (240 sub 3)
   (243 sub 2))
  () 62 0)
 (196 uniq "( move ua - ul , ul ; move ua , 2 ) "
  ((125 sub 3) (157 sub 3) (158 sub 3) (163 sub 3) (191 sub 2) (202 sub 3)
   (203 sub 3) (206 sub 3) (239 sub 2) (240 sub 3))
  ((1333 nul 8)) -1 0)
 (197 equiv "( move ua , ul ; move ua , ua + 4 ) "
  ((64 del 2) (113 sub 3) (131 sub 3) (148 sub 2) (150 sub 2) (155 sub 3)
   (161 sub 3) (165 sub 2) (168 sub 3) (174 sub 1) (178 sub 2) (194 sub 1)
   (205 sub 1) (213 sub 3) (235 sub 3) (236 sub 3) (242 sub 2) (740 ins 4))
  () 64 0)
 (198 equiv "( move ua , ua + ul ; move ua , 4 ) "
  ((63 del 2) (64 del 2) (149 sub 2) (160 sub 3) (163 sub 3) (169 sub 3)
   (171 sub 3) (192 sub 2) (200 sub 3) (204 sub 3) (208 sub 3) (211 sub 3)
   (240 sub 2) (924 ins 5))
  () 64 0)
 (199 equiv "( move ua , ua ; move ua , 4 - ua ) "
  ((63 del 2) (116 sub 3) (123 sub 3) (150 sub 3) (164 sub 3) (177 sub 2)
   (178 sub 3) (194 sub 3) (205 sub 3) (235 sub 2) (236 sub 3) (242 sub 3)
   (406 ins 2) (860 ins 5))
  () 33 0)
 (200 equiv "( move ua , ua + ua ; move 4 , ul ) "
  ((66 del 2) (103 sub 3) (119 sub 2) (128 sub 3) (133 sub 3) (149 sub 3)
   (160 sub 3) (167 sub 3) (176 sub 2) (192 sub 2) (195 sub 3) (198 sub 3)
   (240 sub 3) (243 sub 3))
  () 35 0)
 (201 equiv "( move ua , ua ; move 3 + ul , ul ) "
  ((154 sub 1) (162 sub 3) (166 sub 3) (170 sub 2) (193 sub 3) (207 sub 1)
   (212 sub 3) (230 sub 3) (236 sub 3) (241 sub 3) (645 ins 3) (806 ins 5)
   (919 ins 5))
  () 35 0)
 (202 equiv "( move 3 + ul , ul ; move ua , ua ) "
  ((125 sub 3) (151 sub 3) (157 sub 3) (158 sub 2) (163 sub 3) (191 sub 3)
   (196 sub 3) (206 sub 3))
  () 1 0)
 (203 uniq "( move ua , ul / ua ; move ua , 2 ) "
  ((61 del 2) (149 sub 3) (158 sub 3) (167 sub 2) (191 sub 2) (195 sub 3)
   (196 sub 3) (204 sub 2) (210 sub 3) (239 sub 3) (240 sub 3) (243 sub 1)
   (446 ins 2) (947 ins 5) (985 ins 5))
  () -1 0)
 (204 equiv "( move ua , ul - ua ; move ua , 4 ) "
  ((63 del 2) (64 del 2) (107 sub 3) (128 sub 2) (132 sub 2) (156 sub 3)
   (167 sub 2) (171 sub 3) (175 sub 3) (191 sub 3) (195 sub 2) (198 sub 3)
   (203 sub 2) (208 sub 3) (210 sub 2) (239 sub 3) (240 sub 3) (243 sub 2)
   (657 ins 3) (1002 ins 5))
  () 64 0)
 (205 uniq "( move ua , ul ; move ua , ua - 4 ) "
  ((64 del 2) (114 sub 3) (131 sub 2) (148 sub 3) (150 sub 1) (161 sub 3)
   (164 sub 3) (165 sub 2) (168 sub 3) (173 sub 3) (174 sub 2) (177 sub 3)
   (178 sub 2) (194 sub 2) (197 sub 1) (199 sub 3) (213 sub 2) (235 sub 3)
   (242 sub 2) (447 ins 2))
  () -1 0)
 (206 uniq "( move ua + ul , ul ; move 3 , ul ) "
  ((151 sub 3) (157 sub 3) (163 sub 3) (172 sub 1) (191 sub 3) (196 sub 3)
   (202 sub 3) (209 sub 2) (239 sub 3) (240 sub 3))
  ((1222 nul 8) (1427 nul 9)) -1 0)
 (207 equiv "( move ua , ul ; move 3 + ul , ul ) "
  ((154 sub 1) (161 sub 3) (168 sub 3) (170 sub 1) (193 sub 2) (201 sub 1)
   (919 ins 5) (946 ins 5))
  () 67 0)
 (208 equiv "( move ua , 3 - ul ; move ua , ul ) "
  ((128 sub 3) (129 sub 3) (156 sub 2) (160 sub 2) (169 sub 3) (171 sub 2)
   (175 sub 2) (192 sub 3) (195 sub 2) (198 sub 3) (204 sub 3) (209 sub 3)
   (211 sub 3) (240 sub 2) (243 sub 3))
  () 62 0)
 (209 uniq "( move ua + 3 , ul ; move ul , ul ) "
  ((151 sub 3) (163 sub 3) (172 sub 3) (206 sub 2) (208 sub 3) (442 ins 2)
   (653 ins 3))
  ((442 ins 2) (814 nul 6)) -1 0)
 (210 equiv "( move 4 , ul - ua ; move ua , ua ) "
  ((130 sub 2) (167 sub 3) (175 sub 2) (195 sub 3) (203 sub 3) (204 sub 2)
   (211 sub 2) (243 sub 3))
  () 1 0)
 (211 equiv "( move 4 , ua - ul ; move ua , ua ) "
  ((130 sub 2) (169 sub 3) (171 sub 3) (175 sub 3) (198 sub 3) (208 sub 3)
   (210 sub 2))
  () 2 0)
 (212 uniq "( move 4 , ua ; move ua - ul , ul ) "
  ((115 sub 3) (152 sub 3) (166 sub 2) (201 sub 3) (236 sub 3) (241 sub 3)
   (248 sub 3))
  () -1 0)
 (213 equiv "( move 4 , ul ; move ua , ua - ul ) "
  ((114 sub 3) (116 sub 3) (131 sub 2) (150 sub 3) (173 sub 2) (174 sub 2)
   (177 sub 2) (197 sub 3) (205 sub 2) (242 sub 3) (842 ins 5) (906 ins 5))
  () 131 0)
 (214 equiv "( pen 2 ; pen 3 + 3 ; move 4 , 0 ) " ((269 sub 2)) () 12 0)
 (215 equiv "( pen ua ; move ua + ul , ua + 2 ) "
  ((18 del 4) (70 del 2) (216 sub 3) (218 sub 3)) () 70 0)
 (216 equiv "( pen ua ; move ua + ul , 2 - ua ) "
  ((18 del 4) (70 del 2) (137 sub 2) (145 sub 3) (182 sub 3) (215 sub 3)
   (218 sub 3))
  () 70 0)
 (217 equiv "( pen ua + ua ; move ua + 4 , ul ) "
  ((16 del 4) (77 del 2) (136 sub 2) (142 sub 3)) () 77 0)
 (218 uniq "( pen ua ; move ua + 3 , ua - ul ) "
  ((18 del 4) (137 sub 2) (183 sub 1) (184 sub 2) (215 sub 3) (216 sub 3))
  ((1279 nul 13)) -1 0)
 (219 equiv "( pen 3 + ul ; move ua , ua + ul ) "
  ((82 del 2) (180 sub 2) (181 sub 3) (187 sub 3)) () 17 0)
 (220 equiv "( move ua + 4 , ua + ul ; pen ua ) " ((224 sub 2)) () 1 0)
 (221 equiv "( pen ua - ul ; move ua , ua * 2 ) "
  ((18 del 4) (74 del 2) (144 sub 3) (242 sub 3)) () 14 0)
 (222 equiv "( pen ul / ua ; move ua , 2 - ua ) "
  ((18 del 4) (84 del 2) (96 del 2) (188 sub 3) (225 sub 3) (229 sub 2)
   (231 sub 2) (247 sub 2))
  () 0 0)
 (223 equiv "( move ua + 4 , ul ; pen ul - ua ) " ((151 sub 3)) () 1 0)
 (224 equiv "( move ua + 4 , ua - ul ; pen ul ) " ((220 sub 2)) () 2 0)
 (225 equiv "( pen ul / ua ; move ua , ua + 4 ) "
  ((18 del 4) (96 del 2) (194 sub 3) (222 sub 3) (229 sub 3) (231 sub 3)
   (247 sub 2))
  () 0 0)
 (226 uniq "( pen ua * ua ; move ua , 3 / ua ) " ((18 del 4)) () -1 0)
 (227 uniq "( pen 4 ; move 2 , 0 ; move ua , 2 ) "
  ((259 sub 3) (262 sub 2) (264 sub 3) (276 sub 3) (295 sub 3) (298 sub 2)
   (312 sub 3))
  () -1 0)
 (228 equiv "( pen ul / ua ; move ua - ul , 3 ) "
  ((91 del 2) (229 sub 2) (230 sub 3) (248 sub 1)) () 0 0)
 (229 equiv "( pen ul / ua ; move ua , ul - 3 ) "
  ((76 del 2) (91 del 2) (97 del 2) (222 sub 2) (225 sub 3) (228 sub 2)
   (231 sub 2) (247 sub 3) (248 sub 3))
  () 0 0)
 (230 equiv "( pen ul / ua ; move 3 + ul , ul ) "
  ((78 del 2) (139 sub 3) (188 sub 3) (190 sub 2) (201 sub 3) (228 sub 3)
   (248 sub 2))
  () 0 0)
 (231 equiv "( pen ul / ua ; move ua , 4 - ul ) "
  ((97 del 2) (222 sub 2) (225 sub 3) (229 sub 2) (247 sub 3) (248 sub 3)) ()
  0 0)
 (232 uniq "( pen 2 ; move ul , 0 ; move 3 , 4 ) "
  ((251 sub 3) (276 sub 3) (279 sub 3) (312 sub 3)) () -1 0)
 (233 uniq "( move 2 , 4 ; pen 2 ; move ua , 4 ) "
  ((237 sub 3) (263 sub 3) (281 sub 3) (289 sub 3)) () -1 0)
 (234 uniq "( move ua , 3 ; pen 4 ; move 4 , 2 ) "
  ((237 sub 3) (258 sub 3) (287 sub 3) (292 sub 3) (294 sub 3) (314 sub 3)
   (329 sub 3))
  () -1 0)
 (235 uniq "( move ua , ua ; move ua , ua * ua ) "
  ((122 sub 3) (148 sub 3) (150 sub 3) (159 sub 2) (165 sub 3) (178 sub 3)
   (194 sub 2) (197 sub 3) (199 sub 2) (205 sub 3) (236 sub 3) (242 sub 1)
   (247 sub 3) (483 ins 2) (936 ins 5))
  ((1417 nul 15)) -1 0)
 (236 equiv "( move ua , ua ; move ua , ul + ul ) "
  ((113 sub 3) (148 sub 3) (155 sub 3) (164 sub 3) (194 sub 2) (197 sub 3)
   (199 sub 3) (201 sub 3) (212 sub 3) (235 sub 3) (448 ins 2) (804 ins 4)
   (1011 ins 5))
  () 32 0)
 (237 uniq "( move ua , 4 ; pen 4 ; move 3 , 4 ) "
  ((233 sub 3) (234 sub 3) (277 sub 3) (283 sub 3) (292 sub 3) (329 sub 3))
  () -1 0)
 (238 equiv "( pen 3 ; move 4 , 2 ; v0 = ul ) " ((11 del 5)) () 11 0)
 (239 uniq "( move ua - ul , ua ; move ua , ul ) "
  ((157 sub 3) (158 sub 3) (172 sub 3) (191 sub 2) (195 sub 3) (196 sub 2)
   (203 sub 3) (204 sub 3) (206 sub 3) (240 sub 3) (243 sub 2))
  ((1238 nul 7) (1287 nul 10)) -1 0)
 (240 equiv "( move ua , ul + ul ; move ua , ul ) "
  ((119 sub 3) (149 sub 3) (156 sub 3) (160 sub 2) (169 sub 2) (171 sub 3)
   (192 sub 2) (195 sub 3) (196 sub 3) (198 sub 2) (200 sub 3) (203 sub 3)
   (204 sub 3) (206 sub 3) (208 sub 2) (239 sub 3) (243 sub 2) (670 ins 3)
   (1027 ins 5))
  () 62 0)
 (241 uniq "( move ua , ua ; move ul - ua , ul ) "
  ((112 sub 3) (152 sub 3) (162 sub 3) (201 sub 3) (212 sub 3) (957 ins 5)
   (987 ins 5))
  ((957 ins 5) (987 ins 5) (1344 nul 12) (1474 nul 9)) -1 0)
 (242 uniq "( move ua , ul ; move ua , ua * ua ) "
  ((122 sub 3) (148 sub 3) (150 sub 3) (159 sub 2) (165 sub 3) (174 sub 3)
   (177 sub 3) (178 sub 2) (194 sub 3) (197 sub 2) (199 sub 3) (205 sub 2)
   (213 sub 3) (221 sub 3) (235 sub 1) (483 ins 2))
  ((483 ins 2)) -1 0)
 (243 uniq "( move ua , ul / ua ; move ua , ul ) "
  ((133 sub 3) (156 sub 3) (160 sub 3) (167 sub 3) (175 sub 3) (176 sub 3)
   (191 sub 3) (192 sub 3) (195 sub 2) (200 sub 3) (203 sub 1) (204 sub 2)
   (208 sub 3) (210 sub 3) (239 sub 2) (240 sub 2) (931 ins 5) (947 ins 5))
  ((947 ins 5)) -1 0)
 (244 equiv "( pen 0 ; pen 3 ; move 4 + 2 , ua ) " ((267 sub 3)) () 19 0)
 (245 equiv "( pen 3 ; move 3 * 2 , ua ; pen 0 ) " () () 19 0)
 (246 equiv "( move 0 , ul ; pen ua ; move 4 , 0 ) "
  ((12 del 5) (257 sub 2) (284 sub 3) (285 sub 3) (294 sub 2) (296 sub 3)
   (297 sub 2) (325 sub 3) (351 sub 3))
  () 16 0)
 (247 equiv "( pen ul / ua ; move ua , ua * ua ) "
  ((18 del 4) (96 del 2) (222 sub 2) (225 sub 2) (229 sub 3) (231 sub 3)
   (235 sub 3))
  () 0 0)
 (248 equiv "( pen ul / ua ; move ua - ul , ul ) "
  ((97 del 2) (190 sub 3) (212 sub 3) (228 sub 1) (229 sub 3) (230 sub 2)
   (231 sub 3))
  () 0 0)
 (249 uniq "( move ua , 2 ; pen ua ; move 2 , 3 ) "
  ((260 sub 3) (277 sub 3) (281 sub 3) (283 sub 3) (287 sub 2) (292 sub 3)
   (294 sub 2) (351 sub 3))
  () -1 0)
 (250 equiv "( pen 3 ; move ua , 4 ; move 0 , ua ) "
  ((13 del 5) (251 sub 3) (290 sub 2) (291 sub 3) (300 sub 2) (321 sub 3)) ()
  13 0)
 (251 uniq "( pen 3 ; move ua , 0 ; move 3 , ul ) "
  ((232 sub 3) (250 sub 3) (254 sub 3) (259 sub 3) (276 sub 3) (290 sub 2)
   (291 sub 2) (299 sub 3) (300 sub 2) (312 sub 3) (324 sub 3) (359 sub 3)
   (516 ins 2))
  () -1 0)
 (252 uniq "( pen ua ; move 2 , 3 ; move 2 , ul ) "
  ((259 sub 3) (298 sub 3) (299 sub 3) (312 sub 3) (328 sub 3) (350 sub 3)
   (358 sub 3) (359 sub 3))
  () -1 0)
 (253 uniq "( move 4 , 2 ; move ua , 2 ; pen ua ) " ((274 sub 3) (278 sub 1))
  ((698 nul 10)) -1 0)
 (254 equiv "( pen 3 ; move 0 , ul ; move 4 , ul ) "
  ((251 sub 3) (290 sub 3) (291 sub 3) (298 sub 3) (300 sub 3) (305 sub 3)
   (308 sub 3) (324 sub 2) (328 sub 2) (358 sub 3) (359 sub 3))
  () 11 0)
 (255 uniq "( move 3 , 2 ; pen 3 ; move ua , ul ) "
  ((260 sub 3) (263 sub 2) (292 sub 2) (296 sub 3) (302 sub 3) (323 sub 3))
  () -1 0)
 (256 equiv "( pen 4 ; move ua , ul ; v0 = 0 ) "
  ((17 del 5) (280 sub 2) (293 sub 3) (304 sub 1)) () 17 0)
 (257 equiv "( v0 = ul ; pen ua ; move 4 , 0 ) " ((12 del 5) (246 sub 2)) ()
  12 0)
 (258 uniq "( move ul , 0 ; pen 4 ; move 4 , ul ) "
  ((234 sub 3) (292 sub 3) (296 sub 3) (297 sub 3) (302 sub 3) (323 sub 2)
   (329 sub 3))
  () -1 0)
 (259 uniq "( pen 4 ; move 2 , ua ; move 3 , ul ) "
  ((17 del 5) (227 sub 3) (251 sub 3) (252 sub 3) (262 sub 3) (264 sub 3)
   (265 sub 3) (298 sub 2) (299 sub 3) (305 sub 3) (350 sub 3) (863 ins 4)
   (1141 ins 6))
  () -1 0)
 (260 uniq "( move 3 , 4 ; pen ua ; move 2 , ul ) "
  ((249 sub 3) (255 sub 3) (277 sub 3) (283 sub 2) (287 sub 2) (289 sub 3)
   (296 sub 2) (297 sub 3))
  () -1 0)
 (261 equiv "( pen 4 ; v0 = 0 ; move ua , ul ) " ((17 del 5)) () 17 0)
 (262 uniq "( pen 4 ; move 2 , ul ; move ua , 4 ) "
  ((227 sub 2) (259 sub 3) (264 sub 3) (276 sub 3) (295 sub 2) (298 sub 1)
   (305 sub 3) (322 sub 3) (324 sub 3) (358 sub 3))
  () -1 0)
 (263 uniq "( move 3 , ul ; pen 3 ; move ua , 4 ) "
  ((13 del 5) (233 sub 3) (255 sub 2) (284 sub 3) (285 sub 3) (301 sub 3)) ()
  -1 0)
 (264 uniq "( pen 4 ; move 4 , 2 ; move ua , ul ) "
  ((17 del 5) (40 del 3) (227 sub 3) (259 sub 3) (262 sub 3) (265 sub 1)
   (291 sub 3) (295 sub 3) (298 sub 2) (299 sub 3) (305 sub 2) (312 sub 3)
   (322 sub 3) (324 sub 3) (350 sub 3) (358 sub 3))
  () -1 0)
 (265 uniq "( pen 4 ; move 4 , 2 ; move ul , ul ) "
  ((259 sub 3) (264 sub 1) (291 sub 2) (298 sub 3) (299 sub 2) (300 sub 3)
   (305 sub 1) (359 sub 3))
  () -1 0)
 (266 equiv "( pen 2 + 2 ; move ua , ul ; pen 0 ) " ((306 sub 3)) () 17 0)
 (267 uniq "( pen 0 ; pen 3 ; move ua , 2 / ua ) " ((244 sub 3) (273 sub 2))
  () -1 0)
 (268 equiv "( pen 3 ; move ua , 0 - 4 ; pen ul ) " ((13 del 5) (272 sub 3))
  () 52 0)
 (269 equiv "( pen 2 ; pen ua + 3 ; move 4 , ul ) "
  ((16 del 5) (214 sub 2) (308 sub 2)) () 16 0)
 (270 equiv "( pen 3 ; move ua , 4 ; pen 2 + ul ) "
  ((13 del 5) (300 sub 3) (309 sub 3)) () 13 0)
 (271 equiv "( pen 4 ; ( move ua , ul ; pen 0 ) ) " ((17 del 5)) () 17 0)
 (272 equiv "( pen 3 ; move 4 , 3 - ul ; pen ul ) " ((268 sub 3)) () 11 0)
 (273 uniq "( pen 4 ; pen 3 ; move ua , 4 / ua ) " ((13 del 5) (267 sub 2))
  () -1 0)
 (274 equiv "( move ua , ua ; move ua , 2 ; pen 0 ) "
  ((61 del 3) (253 sub 3) (278 sub 2) (282 sub 2)) () 32 0)
 (275 equiv "( ( pen 4 ; move ua , ul ) ; pen 2 ) " ((17 del 5)) () 17 0)
 (276 uniq "( pen ua ; move ua , 0 ; move ua , 4 ) "
  ((14 del 5) (18 del 5) (33 del 3) (227 sub 3) (232 sub 3) (251 sub 3)
   (262 sub 3) (295 sub 3) (312 sub 2) (320 sub 2) (321 sub 3) (350 sub 3)
   (358 sub 3) (359 sub 3))
  ((1558 nul 10)) -1 0)
 (277 uniq "( move ua , 4 ; pen ua ; move 0 , ua ) "
  ((237 sub 3) (249 sub 3) (260 sub 3) (283 sub 1) (287 sub 3) (289 sub 3)
   (294 sub 3) (325 sub 3) (351 sub 2))
  ((283 sub 1) (1282 nul 8)) -1 0)
 (278 equiv "( move ua , 2 ; move ua , 2 ; pen ua ) "
  ((253 sub 1) (274 sub 2)) () 149 0)
 (279 equiv "( pen ua ; move ul , ul ; move 3 , 0 ) "
  ((232 sub 3) (290 sub 3) (312 sub 3) (321 sub 3) (324 sub 3) (328 sub 2)
   (350 sub 3) (358 sub 2) (359 sub 3))
  () 16 0)
 (280 equiv "( pen ua ; move ua , ua ; v0 = 0 ) "
  ((14 del 5) (18 del 5) (256 sub 2) (293 sub 1) (304 sub 3)) () 14 0)
 (281 uniq "( move ua , ua ; pen 2 ; move ua , 3 ) "
  ((233 sub 3) (249 sub 3) (286 sub 3) (292 sub 3) (294 sub 3) (302 sub 3)
   (314 sub 2) (323 sub 3))
  ((599 nul 3)) -1 0)
 (282 equiv "( move ua , ua ; move 4 , ul ; pen 0 ) "
  ((66 del 3) (274 sub 2) (330 sub 2)) () 35 0)
 (283 equiv "( move ua , 4 ; pen ua ; move 0 , ul ) "
  ((237 sub 3) (249 sub 3) (260 sub 2) (277 sub 1) (287 sub 2) (289 sub 3)
   (292 sub 3) (294 sub 3) (296 sub 3) (297 sub 3) (329 sub 3) (351 sub 3))
  () 277 0)
 (284 equiv "( move 0 , ul ; pen 4 ; move ua , ua ) "
  ((246 sub 3) (263 sub 3) (285 sub 3) (292 sub 3) (301 sub 1) (302 sub 3)
   (323 sub 3) (325 sub 3) (329 sub 3) (351 sub 3))
  () 17 0)
 (285 equiv "( move 4 , ul ; pen ul ; move ua , 0 ) "
  ((246 sub 3) (263 sub 3) (284 sub 3) (286 sub 3) (301 sub 2) (302 sub 3)
   (314 sub 3) (325 sub 3))
  () 1 0)
 (286 uniq "( move 4 , ua ; pen ua ; move ua , 2 ) "
  ((37 del 3) (281 sub 3) (285 sub 3) (289 sub 2) (296 sub 3) (301 sub 3)
   (302 sub 2) (314 sub 2) (323 sub 3) (325 sub 3) (771 ins 4))
  () -1 0)
 (287 uniq "( move ua , 3 ; pen ua ; move 2 , ul ) "
  ((234 sub 3) (249 sub 2) (260 sub 2) (277 sub 3) (283 sub 2) (292 sub 3)
   (294 sub 3) (296 sub 3) (297 sub 3) (329 sub 3) (351 sub 3) (884 ins 4))
  ((884 ins 4)) -1 0)
 (288 uniq "( move ua + 3 , 4 + 2 ; move 3 , 2 ) " ((355 sub 2)) () -1 0)
 (289 uniq "( move ul , 4 ; pen ua ; move ua , 2 ) "
  ((233 sub 3) (260 sub 3) (277 sub 3) (283 sub 3) (286 sub 2) (314 sub 3)
   (323 sub 3) (791 ins 4) (812 ins 4) (1227 ins 6))
  () -1 0)
 (290 uniq "( pen 3 ; move ua , ul ; move 3 , ua ) "
  ((250 sub 2) (251 sub 2) (254 sub 3) (279 sub 3) (291 sub 3) (295 sub 3)
   (300 sub 3) (322 sub 3) (324 sub 3) (359 sub 3) (778 ins 4))
  ((628 nul 9)) -1 0)
 (291 uniq "( pen 3 ; move ua , 2 ; move ul , ul ) "
  ((250 sub 3) (251 sub 2) (254 sub 3) (264 sub 3) (265 sub 2) (290 sub 3)
   (299 sub 2) (300 sub 1) (305 sub 3) (320 sub 3) (324 sub 3) (359 sub 2)
   (950 ins 4))
  () -1 0)
 (292 uniq "( move ua , 2 ; pen 4 ; move ua , ul ) "
  ((17 del 5) (62 del 3) (234 sub 3) (237 sub 3) (249 sub 3) (255 sub 2)
   (258 sub 3) (281 sub 3) (283 sub 3) (284 sub 3) (287 sub 3) (301 sub 3)
   (302 sub 2) (314 sub 3) (323 sub 2) (329 sub 2))
  () -1 0)
 (293 equiv "( pen ua ; move ua , ua ; v0 = 3 ) "
  ((14 del 5) (18 del 5) (256 sub 3) (280 sub 1) (304 sub 3)) () 14 0)
 (294 uniq "( move ua , ul ; pen ua ; move 4 , 3 ) "
  ((234 sub 3) (246 sub 2) (249 sub 2) (277 sub 3) (281 sub 3) (283 sub 3)
   (287 sub 3) (296 sub 3) (297 sub 2) (325 sub 3) (329 sub 3) (351 sub 2)
   (784 ins 4) (1147 ins 6))
  () -1 0)
 (295 uniq "( pen 4 ; move ua , ul ; move ua , 3 ) "
  ((17 del 5) (227 sub 3) (262 sub 2) (264 sub 3) (276 sub 3) (290 sub 3)
   (298 sub 2) (299 sub 3) (305 sub 3) (322 sub 3) (324 sub 3) (358 sub 3)
   (359 sub 3))
  () -1 0)
 (296 uniq "( move 3 , ua ; pen ua ; move 4 , ul ) "
  ((16 del 5) (246 sub 3) (255 sub 3) (258 sub 3) (260 sub 2) (283 sub 3)
   (286 sub 3) (287 sub 3) (294 sub 3) (297 sub 2) (302 sub 3) (323 sub 3))
  () -1 0)
 (297 uniq "( move 2 , ul ; pen ua ; move 4 , ul ) "
  ((16 del 5) (246 sub 2) (258 sub 3) (260 sub 3) (283 sub 3) (287 sub 3)
   (294 sub 2) (296 sub 2) (325 sub 3) (329 sub 3) (351 sub 3) (1156 ins 6))
  () -1 0)
 (298 uniq "( pen 4 ; move 2 , ul ; move ua , ul ) "
  ((17 del 5) (227 sub 2) (252 sub 3) (254 sub 3) (259 sub 2) (262 sub 1)
   (264 sub 2) (265 sub 3) (295 sub 2) (299 sub 3) (305 sub 2) (312 sub 3)
   (322 sub 3) (324 sub 2) (328 sub 3) (350 sub 3) (358 sub 2) (359 sub 3))
  () -1 0)
 (299 uniq "( pen 4 ; move ua , 3 ; move ul , ul ) "
  ((17 del 5) (251 sub 3) (252 sub 3) (259 sub 3) (264 sub 3) (265 sub 2)
   (291 sub 2) (295 sub 3) (298 sub 3) (300 sub 2) (305 sub 2) (320 sub 3)
   (359 sub 2) (887 ins 4))
  () -1 0)
 (300 uniq "( pen 3 ; move ua , 4 ; move ul , ul ) "
  ((13 del 5) (250 sub 2) (251 sub 2) (254 sub 3) (265 sub 3) (270 sub 3)
   (290 sub 3) (291 sub 1) (299 sub 2) (305 sub 3) (320 sub 3) (324 sub 3)
   (359 sub 2))
  () -1 0)
 (301 uniq "( move 4 , ul ; pen 4 ; move ua , ua ) "
  ((263 sub 3) (284 sub 1) (285 sub 2) (286 sub 3) (292 sub 3) (302 sub 2)
   (323 sub 3) (325 sub 2) (329 sub 3) (351 sub 3))
  ((1306 nul 8)) -1 0)
 (302 uniq "( move 4 , ua ; pen 4 ; move ua , ul ) "
  ((17 del 5) (255 sub 3) (258 sub 3) (281 sub 3) (284 sub 3) (285 sub 3)
   (286 sub 2) (292 sub 2) (296 sub 3) (301 sub 2) (314 sub 3) (323 sub 1)
   (329 sub 3))
  () -1 0)
 (303 uniq "( move 3 + ul , 4 ; move 4 + 3 , 2 ) " () () -1 0)
 (304 equiv "( pen 4 ; move ua , ul ; v0 = ua ) "
  ((17 del 5) (256 sub 1) (280 sub 3) (293 sub 3)) () 17 0)
 (305 uniq "( pen 4 ; move 4 , ul ; move ul , ul ) "
  ((254 sub 3) (259 sub 3) (262 sub 3) (264 sub 2) (265 sub 1) (291 sub 3)
   (295 sub 3) (298 sub 2) (299 sub 2) (300 sub 3) (322 sub 3) (324 sub 3)
   (328 sub 3) (358 sub 3) (359 sub 2))
  () -1 0)
 (306 equiv "( pen 3 + ul ; move ua , ul ; pen 2 ) " ((82 del 3) (266 sub 3))
  () 17 0)
 (307 equiv "( pen 3 ; pen ua ; move ua , ul - 3 ) " ((76 del 3)) () 76 0)
 (308 equiv "( pen 3 ; pen ua + ul ; move 4 , ul ) "
  ((16 del 5) (89 del 3) (254 sub 3) (269 sub 2)) () 16 0)
 (309 equiv "( pen 4 ; move ua , ul ; pen 2 - ul ) " ((17 del 5) (270 sub 3))
  () 17 0)
 (310 equiv "( pen ua + 4 ; move 4 , ul ; pen ul ) "
  ((16 del 5) (55 del 3) (330 sub 3)) () 16 0)
 (311 uniq "( move 3 + 2 , 0 ; move ua , ul - 2 ) "
  ((318 sub 3) (319 sub 2) (349 sub 3) (404 sub 3))
  ((404 sub 3) (1182 nul 11)) -1 0)
 (312 uniq "( pen ua ; move ul , 0 ; move ua , ul ) "
  ((227 sub 3) (232 sub 3) (251 sub 3) (252 sub 3) (264 sub 3) (276 sub 2)
   (279 sub 3) (298 sub 3) (321 sub 2) (324 sub 2) (328 sub 2) (350 sub 1)
   (358 sub 1) (359 sub 3))
  ((350 sub 1) (1551 nul 11)) -1 0)
 (313 uniq "( move ua + 2 , ul ; move 3 + 2 , 2 ) " () () -1 0)
 (314 equiv "( move ua , ua ; pen ul ; move ua , 2 ) "
  ((61 del 3) (234 sub 3) (281 sub 2) (285 sub 3) (286 sub 2) (289 sub 3)
   (292 sub 3) (302 sub 3) (323 sub 3))
  () 32 0)
 (315 uniq "( move ua , ua * 2 ; move 3 + 2 , 2 ) " () () -1 0)
 (316 equiv "( pen ua ; move 3 , ua ; move ul , ua ) "
  ((18 del 5) (320 sub 2) (321 sub 3) (350 sub 3) (359 sub 3) (972 ins 4)
   (1139 ins 6))
  () 12 0)
 (317 uniq "( move ua + 4 , 4 ; move 0 - 2 , ul ) " () () -1 0)
 (318 uniq "( move 3 + 2 , ul ; move ua , 4 + 2 ) " ((311 sub 3) (326 sub 3))
  () -1 0)
 (319 uniq "( move 2 + 2 , 4 ; move ua , ul - 2 ) "
  ((311 sub 2) (392 sub 3) (404 sub 3)) () -1 0)
 (320 uniq "( pen ua ; move ua , ua ; move ul , 4 ) "
  ((18 del 5) (276 sub 2) (291 sub 3) (299 sub 3) (300 sub 3) (316 sub 2)
   (350 sub 3) (359 sub 2) (642 ins 2) (1002 ins 4) (1209 ins 6))
  ((642 ins 2)) -1 0)
 (321 uniq "( pen ua ; move ul , 4 ; move ua , ua ) "
  ((15 del 5) (18 del 5) (250 sub 3) (276 sub 3) (279 sub 3) (312 sub 2)
   (316 sub 3) (322 sub 3) (324 sub 3) (328 sub 3) (350 sub 2) (358 sub 2))
  ((958 nul 11) (971 nul 5)) -1 0)
 (322 equiv "( pen ul ; move 4 , ul ; move ua , ua ) "
  ((262 sub 3) (264 sub 3) (290 sub 3) (295 sub 3) (298 sub 3) (305 sub 3)
   (321 sub 3) (324 sub 3) (358 sub 3))
  () 1 0)
 (323 uniq "( move ul , ua ; pen 4 ; move ua , ul ) "
  ((17 del 5) (255 sub 3) (258 sub 2) (281 sub 3) (284 sub 3) (286 sub 3)
   (289 sub 3) (292 sub 2) (296 sub 3) (301 sub 3) (302 sub 1) (314 sub 3)
   (329 sub 3))
  () -1 0)
 (324 uniq "( pen 3 ; move ul , ul ; move ua , ul ) "
  ((251 sub 3) (254 sub 2) (262 sub 3) (264 sub 3) (279 sub 3) (290 sub 3)
   (291 sub 3) (295 sub 3) (298 sub 2) (300 sub 3) (305 sub 3) (312 sub 2)
   (321 sub 3) (322 sub 3) (328 sub 2) (350 sub 2) (358 sub 1) (359 sub 3))
  () -1 0)
 (325 uniq "( move 4 , ul ; pen ua ; move ul , ua ) "
  ((246 sub 3) (277 sub 3) (284 sub 3) (285 sub 3) (286 sub 3) (294 sub 3)
   (297 sub 3) (301 sub 2) (329 sub 3) (351 sub 1) (606 ins 2))
  ((547 nul 3) (585 nul 3) (591 nul 3) (606 ins 2) (892 nul 5)) -1 0)
 (326 uniq "( move 3 + 2 , ul ; move ua , 3 * 3 ) " ((318 sub 3)) () -1 0)
 (327 equiv "( move 4 , 4 * 3 ; move ua , 0 - ua ) " () () 4 0)
 (328 uniq "( pen ua ; move ul , ul ; move 4 , ul ) "
  ((16 del 5) (252 sub 3) (254 sub 2) (279 sub 2) (298 sub 3) (305 sub 3)
   (312 sub 2) (321 sub 3) (324 sub 2) (350 sub 2) (358 sub 1) (359 sub 2)
   (624 ins 2))
  ((624 ins 2) (866 nul 5)) -1 0)
 (329 uniq "( move ua , ul ; pen 4 ; move ul , ul ) "
  ((234 sub 3) (237 sub 3) (258 sub 3) (283 sub 3) (284 sub 3) (287 sub 3)
   (292 sub 2) (294 sub 3) (297 sub 3) (301 sub 3) (302 sub 3) (323 sub 3)
   (325 sub 3) (351 sub 2) (967 ins 4) (978 ins 4))
  ((978 ins 4) (1003 nul 5)) -1 0)
 (330 equiv "( move ua , ul ; move 4 , ul ; pen ul ) "
  ((67 del 3) (282 sub 2) (310 sub 3)) () 67 0)
 (331 uniq "( move 3 + ul , 4 * 3 ; move ua , 2 ) "
  ((363 sub 3) (375 sub 3) (377 sub 3)) () -1 0)
 (332 uniq "( move 4 + 4 , 4 + ul ; move ua , 4 ) "
  ((44 del 4) (355 sub 3) (375 sub 2) (411 sub 2)) () -1 0)
 (333 uniq "( move 4 + 4 , 0 ; move ul , ul / 4 ) " () () -1 0)
 (334 uniq "( move 4 , ul ; move ua + 3 , 4 * 4 ) " ((44 del 4) (373 sub 3))
  () -1 0)
 (335 uniq "( move 4 , 3 / ua ; move ua , 4 + 4 ) " ((336 sub 3)) () -1 0)
 (336 uniq "( move 4 , ul / 4 ; move ua , 4 + 3 ) " ((44 del 4) (335 sub 3))
  () -1 0)
 (337 equiv "( v0 = 4 ; pen 3 ; move ua , v0 ) " () () 13 0)
 (338 equiv "( pen ul ; pen ul / ua ; move ua , 0 ) "
  ((14 del 5) (71 del 3) (385 sub 1)) () 0 0)
 (339 equiv "( pen ua ; pen ua ; move ua + 4 , ul ) " ((16 del 5) (77 del 3))
  () 77 0)
 (340 equiv "( pen ul / ua ; move ua , ul ; pen 3 ) " ((91 del 3) (97 del 3))
  () 0 0)
 (341 uniq "( pen ua - ul ; move 4 + 4 , 3 + 3 ) " () () -1 0)
 (342 uniq "( move ua , 3 + 2 ; move 0 - ua , ul ) " ((62 del 4) (160 del 2))
  () -1 0)
 (343 uniq "( move ua - 2 , ua ; move ua + 2 , 3 ) " ((438 sub 3)) () -1 0)
 (344 equiv "( move ua + 3 , ul - 2 ; move ul , 0 ) " ((368 sub 2)) () 2 0)
 (345 uniq "( move ul - 4 , 0 ; move ua , ua + 2 ) " ((20 del 4) (433 sub 2))
  () -1 0)
 (346 equiv "( move 0 , ua ; move ua + 4 , 2 - ul ) "
  ((402 sub 3) (443 sub 3) (449 sub 3)) () 1 0)
 (347 uniq "( move 4 , 0 ; move ua - ul , 0 - ul ) "
  ((389 sub 3) (449 sub 3)) ((1411 nul 9)) -1 0)
 (348 uniq "( move 0 - 4 , ua - ul ; move ua , 2 ) " ((37 del 4) (439 sub 3))
  () -1 0)
 (349 equiv "( move 3 + ul , 0 ; move ua , 3 - ul ) "
  ((311 sub 3) (441 sub 2)) () 20 0)
 (350 equiv "( pen ua ; move ul , ua ; move ua , ul ) "
  ((252 sub 3) (259 sub 3) (264 sub 3) (276 sub 3) (279 sub 3) (298 sub 3)
   (312 sub 1) (316 sub 3) (320 sub 3) (321 sub 2) (324 sub 2) (328 sub 2)
   (358 sub 1) (359 sub 3) (956 ins 4) (1008 ins 4) (1009 ins 4))
  () 312 0)
 (351 uniq "( move ua , ul ; pen ua ; move ul , ua ) "
  ((246 sub 3) (249 sub 3) (277 sub 2) (283 sub 3) (284 sub 3) (287 sub 3)
   (294 sub 2) (297 sub 3) (301 sub 3) (325 sub 1) (329 sub 2) (675 ins 2)
   (953 ins 4))
  ((1326 nul 8) (1541 nul 9) (1559 nul 10)) -1 0)
 (352 uniq "( move ul , ua - 4 ; move ua + 4 , 0 ) " ((419 sub 2)) () -1 0)
 (353 equiv "( move ua , ul ; move 2 + 2 , ua * 3 ) " () () 1 0)
 (354 uniq "( move ua , 3 ; move ua + 3 , 3 + ul ) " ((36 del 4) (487 sub 3))
  () -1 0)
 (355 uniq "( move ua + 3 , 4 + ul ; move ua , 2 ) "
  ((288 sub 2) (332 sub 3) (375 sub 3) (393 sub 3) (411 sub 3) (439 sub 3)
   (440 sub 3) (442 sub 3))
  () -1 0)
 (356 uniq "( move ua + 3 , 2 ; move ua , ul - 3 ) "
  ((62 del 4) (164 del 2) (395 sub 3)) () -1 0)
 (357 uniq "( move ua - 2 , 2 ; move ua , ua - 4 ) " ((380 sub 3)) () -1 0)
 (358 uniq "( pen ua ; move ul , ul ; move ua , ul ) "
  ((252 sub 3) (254 sub 3) (262 sub 3) (264 sub 3) (276 sub 3) (279 sub 2)
   (295 sub 3) (298 sub 2) (305 sub 3) (312 sub 1) (321 sub 2) (322 sub 3)
   (324 sub 1) (328 sub 1) (350 sub 1) (359 sub 2) (648 ins 2) (674 ins 2)
   (1024 ins 4))
  ((648 ins 2)) -1 0)
 (359 uniq "( pen ua ; move ua , ul ; move ul , ul ) "
  ((251 sub 3) (252 sub 3) (254 sub 3) (265 sub 3) (276 sub 3) (279 sub 3)
   (290 sub 3) (291 sub 2) (295 sub 3) (298 sub 3) (299 sub 2) (300 sub 2)
   (305 sub 2) (312 sub 3) (316 sub 3) (320 sub 2) (324 sub 3) (328 sub 2)
   (350 sub 3) (358 sub 2) (673 ins 2))
  ((673 ins 2) (1555 nul 9)) -1 0)
 (360 equiv "( move ua , ul ; move 3 + ul , 2 * 2 ) " ((434 sub 3)) () 43 0)
 (361 uniq "( move ua + ul , 2 ; move 4 , ua + 4 ) "
  ((21 del 4) (369 sub 3) (376 sub 3)) () -1 0)
 (362 uniq "( move 3 + 2 , ul ; move ua + ul , 4 ) "
  ((366 sub 3) (405 sub 3)) () -1 0)
 (363 uniq "( move 3 + ul , 4 + 2 ; move ua , ul ) "
  ((331 sub 3) (367 sub 3) (375 sub 2) (411 sub 2) (452 sub 3)) () -1 0)
 (364 equiv "( move ua , ua * 3 ; move ua , 4 - 2 ) "
  ((39 del 4) (61 del 4) (63 del 4) (374 sub 3)) () 32 0)
 (365 uniq "( move ua + 2 , ul / ua ; move 2 , 4 ) " ((446 sub 3)) () -1 0)
 (366 uniq "( move ua + 2 , ul ; move ua - 4 , 4 ) "
  ((43 del 4) (64 del 4) (163 del 2) (362 sub 3)) () -1 0)
 (367 uniq "( move ua + ul , 4 - 2 ; move ua , 4 ) "
  ((41 del 4) (363 sub 3) (375 sub 3) (439 sub 3)) () -1 0)
 (368 equiv "( move ua + 4 , ul - 2 ; move ul , 3 ) " ((344 sub 2)) () 2 0)
 (369 uniq "( move 4 + ul , 3 ; move ua , ua + 4 ) "
  ((361 sub 3) (412 sub 3)) () -1 0)
 (370 uniq "( move 3 + 2 , ua / 3 ; move ua , ul ) " ((452 sub 3)) () -1 0)
 (371 uniq "( move 0 - 4 , 3 / ua ; move ua , ul ) " () () -1 0)
 (372 uniq "( move ua - ul , 4 ; move ua , 3 * 2 ) " () () -1 0)
 (373 uniq "( move 4 , ul ; move ua + ul , 2 - 4 ) "
  ((44 del 4) (334 sub 3) (390 sub 3) (449 sub 3)) () -1 0)
 (374 uniq "( move 4 , ua / 3 ; move ua , ua - 2 ) "
  ((37 del 4) (364 sub 3) (394 sub 3)) () -1 0)
 (375 uniq "( move 3 + ul , 4 + ul ; move ua , 4 ) "
  ((331 sub 3) (332 sub 2) (355 sub 3) (363 sub 2) (367 sub 3) (411 sub 2)
   (440 sub 3))
  () -1 0)
 (376 uniq "( move ua + ul , 2 ; move 3 , ul / 4 ) " ((361 sub 3)) () -1 0)
 (377 uniq "( move ua + ul , 4 * 3 ; move 3 , ul ) "
  ((172 del 2) (331 sub 3)) () -1 0)
 (378 uniq "( move ua , 3 / 4 ; move ua - ul , 2 ) "
  ((409 sub 3) (414 sub 3)) () -1 0)
 (379 uniq "( move ul , 3 / 4 ; move ua + 3 , ua ) " ((419 sub 3))
  ((709 nul 8)) -1 0)
 (380 uniq "( move 4 * 2 , ul ; move ua , ua - 4 ) " ((44 del 4) (357 sub 3))
  () -1 0)
 (381 uniq "( move ua , ua ; move ul - 4 , 2 / 4 ) " () () -1 0)
 (382 uniq "( move ua , 3 / ua ; move 4 , ul / 2 ) " ((42 del 4) (66 del 4))
  () -1 0)
 (383 equiv "( v0 = ua ; pen 4 ; move v0 , ul ) " ((421 sub 2)) () 17 0)
 (384 equiv "( pen 4 + ul ; move ua + 2 , ua * 2 ) " ((80 del 4)) () 69 0)
 (385 equiv "( pen ul ; pen ul / ua ; move ua , ua ) "
  ((18 del 5) (96 del 3) (338 sub 1)) () 0 0)
 (386 uniq "( pen 4 + 4 ; move ua - ul , 4 / ua ) " () () -1 0)
 (387 uniq "( pen 3 ; move 0 - 3 , 3 ; move 3 , 4 ) " () () -1 0)
 (388 uniq "( move ua + ul , 0 ; move ua - ul , 2 ) "
  ((32 del 4) (405 sub 3) (438 sub 3)) () -1 0)
 (389 uniq "( move ua , ul ; move ua - ul , 0 - 2 ) "
  ((347 sub 3) (390 sub 3) (434 sub 3)) () -1 0)
 (390 uniq "( move ua , ua ; move ua + ul , 0 - 4 ) "
  ((63 del 4) (373 sub 3) (389 sub 3) (443 sub 3) (449 sub 3)) () -1 0)
 (391 equiv "( move ua + 4 , ua - ul ; move 0 , ua ) "
  ((151 del 2) (410 sub 2) (439 sub 3)) () 2 0)
 (392 equiv "( move 2 + ul , ua ; move ua , ua - 2 ) "
  ((319 sub 3) (404 sub 2) (441 sub 3)) () 114 0)
 (393 uniq "( move ua + ul , ua + 4 ; move ua , 2 ) "
  ((61 del 4) (191 del 2) (355 sub 3) (439 sub 2) (440 sub 3) (446 sub 3)) ()
  -1 0)
 (394 equiv "( move ua , ua / 2 ; move ua , ua + 2 ) "
  ((61 del 4) (374 sub 3) (429 sub 2) (448 sub 2)) () 164 0)
 (395 equiv "( move ua , ua + 2 ; move ua , ul - 3 ) "
  ((62 del 4) (164 del 2) (192 del 2) (356 sub 3) (448 sub 3)) () 164 0)
 (396 equiv "( move ua , 0 - ua ; move ua / 2 , ul ) " () () 152 0)
 (397 uniq "( move 2 , ua + ua ; move ua + 4 , ul ) " ((419 sub 3))
  ((1382 nul 12)) -1 0)
 (398 uniq "( move ua + 2 , 3 ; move ua , ua * ua ) " () () -1 0)
 (399 uniq "( move ua , 3 - ua ; move ul - ua , 2 ) " ((61 del 4)) () -1 0)
 (400 uniq "( move ua + 3 , ul ; move ua , 3 + ul ) " ((441 sub 3)) () -1 0)
 (401 uniq "( move ua - ul , 3 ; move ul - ua , 2 ) " () () -1 0)
 (402 equiv "( move ua , ua ; move ua / 2 , 2 - ul ) "
  ((61 del 4) (346 sub 3)) () 152 0)
 (403 uniq "( move ua + ul , 2 - ua ; move 4 , ul ) " ((66 del 4))
  ((1193 nul 10)) -1 0)
 (404 equiv "( move 4 + ul , ua ; move ua , ul - 2 ) "
  ((37 del 4) (311 sub 3) (319 sub 3) (392 sub 2) (412 sub 3) (441 sub 3)) ()
  311 0)
 (405 uniq "( move ua + ul , 4 ; move ua + ul , 4 ) "
  ((41 del 4) (362 sub 3) (388 sub 3) (438 sub 3)) () -1 0)
 (406 equiv "( move ua , 4 - ua ; move ua , 4 - ua ) "
  ((41 del 4) (63 del 4) (199 del 2)) () 41 0)
 (407 uniq "( move ul - ua , ul ; move ua , 4 * 2 ) "
  ((64 del 4) (413 sub 3) (433 sub 3)) () -1 0)
 (408 uniq "( move ua , ua - ul ; move 4 + ul , 4 ) "
  ((43 del 4) (416 sub 3) (435 sub 3)) () -1 0)
 (409 uniq "( move ua , 3 / ua ; move ua + ul , 4 ) "
  ((39 del 4) (63 del 4) (378 sub 3) (414 sub 3)) () -1 0)
 (410 equiv "( move ua + 4 , ua - ul ; v0 = ua ) " ((391 sub 2)) () 2 0)
 (411 uniq "( move 4 + ul , 4 + ul ; move ua , ul ) "
  ((332 sub 2) (355 sub 3) (363 sub 2) (375 sub 2) (440 sub 3)) () -1 0)
 (412 equiv "( move 4 + ul , 4 ; move ua , ul + ul ) "
  ((369 sub 3) (404 sub 3)) () 125 0)
 (413 uniq "( move ul - ua , 4 ; move ua , ul / 2 ) "
  ((407 sub 3) (433 sub 3)) () -1 0)
 (414 uniq "( move ua , ua / 3 ; move ua - ul , 4 ) "
  ((39 del 4) (63 del 4) (378 sub 3) (409 sub 3)) () -1 0)
 (415 uniq "( move ua , ul / 2 ; move 4 + ul , ul ) "
  ((67 del 4) (416 sub 2) (418 sub 3)) () -1 0)
 (416 uniq "( move ua , ul - 4 ; move 4 + ul , ul ) "
  ((45 del 4) (67 del 4) (408 sub 3) (415 sub 2) (450 sub 3)) () -1 0)
 (417 uniq "( move ua , 4 ; move ul - ua , 4 - ul ) " ((41 del 4)) () -1 0)
 (418 uniq "( move 3 , ul / 3 ; move ua + ul , ul ) " ((415 sub 3)) () -1 0)
 (419 uniq "( move ul , ua / 4 ; move ua + 4 , ul ) "
  ((352 sub 2) (379 sub 3) (397 sub 3)) () -1 0)
 (420 equiv "( pen ua ; v0 = ua ; move ua , v0 ) " ((14 del 6)) () 14 0)
 (421 equiv "( v0 = ua ; pen ua ; move v0 , ua ) " ((383 sub 2)) () 14 0)
 (422 uniq "( move 4 , 4 ; move ua , 4 ; move 0 , 3 ) "
  ((497 sub 3) (498 sub 3))
  ((930 nul 10) (944 nul 9) (1033 nul 8) (1037 nul 10)) -1 0)
 (423 equiv "( pen 4 + 3 ; move ua , 0 ; move 0 , 0 ) " () () 14 0)
 (424 uniq "( pen 3 ; move 2 , 0 ; move 2 , ua + 2 ) " ((472 sub 3)) () -1 0)
 (425 equiv "( pen ua * 4 ; move 3 + ul , ua + ul ) " ((78 del 4)) () 16 0)
 (426 uniq "( move 2 , 3 / ua ; pen 2 ; move 3 , 0 ) " () () -1 0)
 (427 uniq "( move 0 - 4 , 2 ; pen ua ; move 3 , 3 ) " ((613 sub 3)) () -1 0)
 (428 uniq "( move 3 , 0 ; pen 3 * 2 ; move 2 , ul ) " () () -1 0)
 (429 equiv "( move ua , ua + ua ; move ua , ua + 2 ) "
  ((61 del 4) (394 sub 2) (433 sub 3) (483 sub 3)) () 32 0)
 (430 equiv "( move ua , 3 ; move 4 , 3 - 2 ; pen 2 ) " () () 42 0)
 (431 uniq "( move 0 - ua , ul ; move ua , ua + ul ) "
  ((433 sub 3) (436 sub 3)) () -1 0)
 (432 uniq "( pen ua ; move 2 , 4 ; move 2 , 3 - 4 ) " ((27 del 5)) () -1 0)
 (433 uniq "( move ul - ua , ua ; move ua , ua + 2 ) "
  ((61 del 4) (345 sub 2) (407 sub 3) (413 sub 3) (429 sub 3) (431 sub 3))
  ((1070 nul 9)) -1 0)
 (434 equiv "( move ua , ul ; move ua + ul , ua + 2 ) "
  ((193 del 2) (360 sub 3) (389 sub 3) (487 sub 2)) () 193 0)
 (435 equiv "( move ua , ua + ul ; move ua + ul , 2 ) "
  ((61 del 4) (193 del 2) (408 sub 3) (1050 ins 4)) () 193 0)
 (436 uniq "( move 0 - ua , ul ; move ua + ul , ul ) "
  ((431 sub 3) (451 sub 3)) () -1 0)
 (437 equiv "( move ua , 2 ; move ua , 2 ; move 0 , 0 ) " ((491 sub 3)) ()
  149 0)
 (438 uniq "( move ua - ul , ua ; move ua + ul , 2 ) "
  ((61 del 4) (343 sub 3) (388 sub 3) (405 sub 3) (451 sub 3)) () -1 0)
 (439 uniq "( move ua + ul , ua - ul ; move ua , 2 ) "
  ((61 del 4) (191 del 2) (348 sub 3) (355 sub 3) (367 sub 3) (391 sub 3)
   (393 sub 2) (440 sub 3) (446 sub 3))
  () -1 0)
 (440 equiv "( move ua + ul , ul + ul ; move ua , 3 ) "
  ((355 sub 3) (375 sub 3) (393 sub 3) (411 sub 3) (439 sub 3) (446 sub 3))
  () 157 0)
 (441 equiv "( move ua + ul , ua ; move ua , 3 - ul ) "
  ((349 sub 2) (392 sub 3) (400 sub 3) (404 sub 3)) () 191 0)
 (442 equiv "( move ua + 3 , ua + ul ; move ul , ul ) "
  ((209 del 2) (355 sub 3)) () 209 0)
 (443 uniq "( move 3 , ua ; move ua + ul , ua - ul ) "
  ((346 sub 3) (390 sub 3) (449 sub 1)) ((909 nul 8)) -1 0)
 (444 uniq "( pen 4 ; ( move 2 , ua ; move 2 , 3 ) ) " () () -1 0)
 (445 uniq "( pen 4 ; move 2 , 4 ; move 4 , ul - 3 ) " () () -1 0)
 (446 uniq "( move ua + ul , ul / ua ; move ua , 2 ) "
  ((61 del 4) (191 del 2) (203 del 2) (365 sub 3) (393 sub 3) (439 sub 3)
   (440 sub 3) (452 sub 2))
  () -1 0)
 (447 uniq "( move ua , ul + ul ; move ua , ua - 4 ) "
  ((64 del 4) (205 del 2) (483 sub 3)) () -1 0)
 (448 equiv "( move ua , ua / 2 ; move ua , ul + ul ) "
  ((62 del 4) (236 del 2) (394 sub 2) (395 sub 3)) () 164 0)
 (449 uniq "( move 4 , ua ; move ua + ul , ua - ul ) "
  ((346 sub 3) (347 sub 3) (373 sub 3) (390 sub 3) (443 sub 1))
  ((1160 nul 11)) -1 0)
 (450 equiv "( move ua , ul - ua ; move 4 , ul + ul ) "
  ((66 del 4) (67 del 4) (416 sub 3)) () 38 0)
 (451 equiv "( move 4 - ul , ul ; move ua + ul , ua ) "
  ((436 sub 3) (438 sub 3)) () 1 0)
 (452 uniq "( move 3 + ul , ul / ua ; move ua , ul ) "
  ((363 sub 3) (370 sub 3) (446 sub 2)) ((1028 nul 10)) -1 0)
 (453 uniq "( pen 2 ; move ua + 2 , 3 * 3 ; pen 2 ) " () () -1 0)
 (454 uniq "( move ua , 3 ; move 2 , 3 ; move 3 , ul ) "
  ((552 sub 3) (563 sub 3) (629 sub 3)) () -1 0)
 (455 uniq "( move 4 , 0 ; move 3 , ul ; move ua , 4 ) "
  ((44 del 5) (456 sub 3) (507 sub 3)) () -1 0)
 (456 uniq "( move ua , 2 ; move 3 , 4 ; move ua , 4 ) "
  ((21 del 5) (39 del 5) (41 del 5) (455 sub 3) (564 sub 3)) () -1 0)
 (457 equiv "( move 0 , 0 ; pen ua ; move ua , 0 - 3 ) " () () 87 0)
 (458 equiv "( pen ua + ul ; move 3 + ul , ul - ua ) " ((78 del 4)) () 16 0)
 (459 uniq "( move 4 , ua ; pen ua ; move 2 + 2 , 0 ) " () ((486 nul 5)) -1
  0)
 (460 equiv "( pen ua ; v0 = 2 ; move ua + 2 , 0 ) " () () 69 0)
 (461 uniq "( pen 3 + ul ; move 3 , 2 ; move ua , 0 ) "
  ((517 sub 3) (581 sub 3) (583 sub 3) (593 sub 3))
  ((480 nul 9) (846 nul 10)) -1 0)
 (462 equiv "( pen 3 ; move 3 , ua ; move 2 + ul , 0 ) " ((503 sub 3)) () 19
  0)
 (463 uniq "( move 3 + 2 , 0 ; pen ua ; move 4 , ul ) " () () -1 0)
 (464 uniq "( move 4 , 0 ; pen 2 ; move 3 , ua - ul ) " ((639 sub 3)) () -1
  0)
 (465 equiv "( pen ua ; move 4 , ua * 3 ; move 0 , 3 ) " () () 12 0)
 (466 uniq "( pen 4 ; move ua , 2 ; move ul , 3 + 2 ) " () () -1 0)
 (467 uniq "( move ua , 3 ; pen 2 ; move 3 , ul - 2 ) " () () -1 0)
 (468 uniq "( move 2 , 0 ; pen ua ; move ua / 3 , 4 ) " () () -1 0)
 (469 uniq "( move 3 , 2 ; pen 2 + ul ; move ua , 4 ) " ((48 del 5)) () -1 0)
 (470 uniq "( move 4 , 2 - 4 ; pen ua ; move ul , 0 ) "
  ((541 sub 3) (585 sub 3)) () -1 0)
 (471 uniq "( move 2 , 4 - 3 ; pen ua ; move ua , 2 ) " () ((1561 nul 11)) -1
  0)
 (472 uniq "( pen 3 ; move 3 , ul ; move 2 , ua + 4 ) " ((424 sub 3)) () -1
  0)
 (473 uniq "( pen 3 ; move 4 + ul , 0 ; move ul , 4 ) " () () -1 0)
 (474 uniq "( move 2 , ua ; pen 2 * 2 ; move 4 , ul ) " ((651 sub 3)) () -1
  0)
 (475 uniq "( move 0 - 4 , 4 ; pen 3 ; move ua , ul ) " ((523 sub 1)) () -1
  0)
 (476 equiv "( ( move 0 , 0 ; pen 4 ) ; move ua , ul ) " () () 17 0)
 (477 uniq "( move 3 + ul , ua ; pen 4 ; move 2 , 4 ) " () () -1 0)
 (478 uniq "( pen 3 ; move 4 , ul ; move ua , 3 + 3 ) " ((532 sub 2)) () -1
  0)
 (479 uniq "( pen 3 ; move 2 , ua - 4 ; move ua , 4 ) "
  ((601 sub 3) (650 sub 3)) () -1 0)
 (480 equiv "( pen 4 ; move 0 , ul ; move 3 * 3 , ul ) " () () 461 0)
 (481 uniq "( pen 4 ; move 2 , ul / ua ; move 3 , 2 ) " () () -1 0)
 (482 uniq "( move 4 , 2 ; pen 4 ; move 3 - ua , ul ) " ((40 del 5)) () -1 0)
 (483 equiv "( move ua , ua + ul ; move ua , ua * ua ) "
  ((235 del 2) (242 del 2) (429 sub 3) (447 sub 3)) () 242 0)
 (484 uniq "( move ul , ul ; pen 4 ; move 4 * 2 , 2 ) " ((549 sub 3)) () -1
  0)
 (485 uniq "( move ul , 2 + ul ; pen 4 ; move 4 , 4 ) " () () -1 0)
 (486 equiv "( ( move 4 , ua ; pen ua ) ; move 4 , 0 ) " () () 459 0)
 (487 equiv "( move ua , ul ; move ua + ul , ul + ul ) "
  ((354 sub 3) (434 sub 2) (509 sub 3) (804 ins 2)) () 193 0)
 (488 uniq "( pen 4 ; move 4 , ul ; move 2 , ul / 2 ) " () () -1 0)
 (489 uniq "( pen 4 ; move ua - 4 , 4 ; move 3 , ul ) " () () -1 0)
 (490 equiv "( ( pen ua ; move 4 , 0 ) ; v0 = ul ) " () () 12 0)
 (491 uniq "( move ua , 3 ; move ul , 2 ; move ua , 0 ) "
  ((437 sub 3) (586 sub 3) (626 sub 2) (629 sub 3)) () -1 0)
 (492 uniq "( move ua , 4 ; ( pen 3 ; move ua , 4 ) ) "
  ((39 del 5) (41 del 5) (537 sub 2)) () -1 0)
 (493 uniq "( ( move 4 , ul ; pen 2 ) ; move 3 , ul ) " () () -1 0)
 (494 equiv "( move 4 , ua ; move 0 , 2 ; move ua , ua ) "
  ((498 sub 3) (552 sub 3) (626 sub 3)) () 20 0)
 (495 equiv "( move ua , 4 ; ( move ua , 4 ; pen 4 ) ) " ((41 del 5)) () 41
  0)
 (496 uniq "( ( pen ua ; move 4 , 4 ) ; move 3 , ul ) " ((624 sub 3)) () -1
  0)
 (497 equiv "( move 4 , 3 ; move ua , ul ; move 0 , ul ) "
  ((422 sub 3) (632 sub 3)) () 169 0)
 (498 uniq "( move 4 , ua ; move ua , 4 ; move ua , 2 ) "
  ((37 del 5) (422 sub 3) (494 sub 3) (499 sub 3) (564 sub 3)) () -1 0)
 (499 uniq "( move 3 , 4 ; move ua , ul ; move ua , 2 ) "
  ((34 del 5) (498 sub 3) (507 sub 3)) () -1 0)
 (500 equiv "( move ua , ul ; v0 = 0 ; move 4 , ul ) "
  ((35 del 5) (67 del 5) (505 sub 2) (559 sub 3)) () 67 0)
 (501 equiv "( ( pen ua ; move 4 , 0 ) ; pen ua - 2 ) " () () 12 0)
 (502 equiv "( move 4 , ua ; move ua , 2 ; v0 = ul ) "
  ((37 del 5) (506 sub 3) (566 sub 2) (571 sub 3)) () 20 0)
 (503 equiv "( pen ua ; move ul , 0 ; move 2 + ul , 0 ) " ((462 sub 3)) () 12
  0)
 (504 equiv "( ( pen ua * 3 ; move ua , 0 ) ; pen 4 ) " () () 14 0)
 (505 equiv "( move ua , ua ; v0 = 3 ; move 4 , ul ) "
  ((35 del 5) (42 del 5) (66 del 5) (500 sub 2) (552 sub 3) (565 sub 3)) ()
  35 0)
 (506 equiv "( move ua , ul ; move 4 , 2 ; v0 = ul ) "
  ((38 del 5) (67 del 5) (502 sub 3) (508 sub 3) (566 sub 3) (571 sub 2)
   (574 sub 1))
  () 38 0)
 (507 uniq "( move 3 , ul ; move 4 , ul ; move ua , 4 ) "
  ((44 del 5) (455 sub 3) (499 sub 3)) () -1 0)
 (508 equiv "( move ua , ul ; move ua , 4 ; v0 = 4 ) "
  ((64 del 5) (506 sub 3) (562 sub 3) (566 sub 2) (574 sub 3)) () 64 0)
 (509 equiv "( pen ua - ul ; move ua - ul , ul + ul ) " ((487 sub 3)) () 72
  0)
 (510 equiv "( move 4 , ua ; move ua , ua + 2 ; pen 0 ) "
  ((37 del 5) (635 sub 2)) () 20 0)
 (511 equiv "( v0 = ua ; pen v0 ; move ua , v0 ) " () () 14 0)
 (512 equiv "( v0 = ua ; pen v0 ; move v0 , ua ) " ((1404 ins 6)) () 14 0)
 (513 uniq "( pen ua ; move 3 + 2 , ua ; move 2 , ul ) " () () -1 0)
 (514 uniq "( pen ua ; move 4 , 0 ; move 2 , ul - ua ) " () () -1 0)
 (515 uniq "( move 2 , 4 ; pen ua ; move ul + ul , 0 ) " ((595 sub 3)) () -1
  0)
 (516 uniq "( pen 3 ; move ua , 0 ; move 3 , ul + ul ) "
  ((49 del 5) (251 del 2)) () -1 0)
 (517 uniq "( pen 3 + 2 ; move ua , 2 ; move ua , ul ) "
  ((62 del 5) (461 sub 3) (579 sub 1) (593 sub 3) (646 sub 3) (648 sub 3))
  ((579 sub 1) (670 nul 6)) -1 0)
 (518 uniq "( pen ua - ul ; move ua , 0 ; move 3 , 4 ) " ((68 del 5)) () -1
  0)
 (519 equiv "( pen ua ; move ua + 2 , ua ; v0 = 2 ) " ((69 del 5)) () 69 0)
 (520 equiv "( pen ul ; move 3 , 0 ; move ua , ul - 3 ) "
  ((649 sub 2) (652 sub 3)) () 114 0)
 (521 equiv "( pen 2 + ul ; move 4 , ua ; move 2 , ua ) " ((24 del 5)) () 19
  0)
 (522 uniq "( move ua , 3 ; pen 4 ; move ua , 0 - ul ) "
  ((599 sub 3) (616 sub 3)) () -1 0)
 (523 uniq "( move 0 - 4 , ua ; pen 3 ; move ua , ul ) "
  ((475 sub 1) (613 sub 3)) () -1 0)
 (524 uniq "( pen ul ; move ua , 4 + 2 ; move ua , 3 ) " () ((1233 nul 10))
  -1 0)
 (525 uniq "( pen ua + ua ; move 2 , 3 ; move 4 , ul ) " ((644 sub 3)) () -1
  0)
 (526 uniq "( pen ua * 3 ; move 3 , ua ; move ua , 2 ) "
  ((34 del 5) (581 sub 2) (611 sub 3)) ((581 sub 2) (1516 nul 10)) -1 0)
 (527 equiv "( move ua , 3 ; pen 2 - ul ; move ua , 3 ) "
  ((36 del 5) (610 sub 3)) () 36 0)
 (528 uniq "( move ua , 4 + 3 ; move ua , 2 ; pen ul ) " () () -1 0)
 (529 uniq "( move 2 , 3 ; pen ua + ul ; move 3 , ul ) "
  ((535 sub 3) (590 sub 3)) () -1 0)
 (530 uniq "( pen ua ; move 3 , 4 ; move 3 , ua + ul ) " ((811 ins 2))
  ((811 ins 2)) -1 0)
 (531 uniq "( move ua , 0 ; pen 3 ; move ua / 4 , ul ) "
  ((35 del 5) (42 del 5) (592 sub 3)) () -1 0)
 (532 uniq "( pen 3 ; move 4 , ua ; move ua , 3 + ul ) "
  ((49 del 5) (478 sub 2) (652 sub 3)) () -1 0)
 (533 uniq "( pen ua * 4 ; move ua , ua ; move 3 , 3 ) " () () -1 0)
 (534 uniq "( move ua , ua ; pen 4 ; move ua , 3 / 2 ) " ((61 del 5)) () -1
  0)
 (535 uniq "( move ul , 3 ; pen 2 + ul ; move 4 , ul ) "
  ((529 sub 3) (621 sub 3)) () -1 0)
 (536 uniq "( pen 2 ; move ul , 4 * 2 ; move ua , ul ) " ((50 del 5)) () -1
  0)
 (537 uniq "( move ua , ua ; ( pen 3 ; move ua , 2 ) ) "
  ((61 del 5) (492 sub 2)) () -1 0)
 (538 equiv "( pen ua ; move 4 , ul - 2 ; v0 = ua ) " () () 27 0)
 (539 uniq "( pen 4 ; move ua , ua * 4 ; move 2 , ul ) " () () -1 0)
 (540 uniq "( move 4 , ul ; pen ua ; move 4 - 2 , ul ) " ((549 sub 3))
  ((962 nul 4)) -1 0)
 (541 uniq "( move 4 , 2 - ul ; pen ua ; move 4 , ul ) " ((470 sub 3)) () -1
  0)
 (542 equiv "( pen 4 + 3 ; v0 = ul ; move ua , ua ) " ((80 del 5)) () 14 0)
 (543 uniq "( move 4 + 4 , ul - 2 ; move 3 * 2 , 2 ) " () () -1 0)
 (544 uniq "( move 3 * 2 , 3 * 2 ; move ua , 2 - 4 ) " () () -1 0)
 (545 equiv "( pen ua ; move 3 + ul , ul ; v0 = 4 ) " ((78 del 5)) () 16 0)
 (546 equiv "( pen 3 + ul ; move ua , ul ; v0 = 4 ) " ((82 del 5)) () 17 0)
 (547 equiv "( move 4 , ul ; ( pen ua ; move ul , 0 ) ) " ((892 ins 2)) ()
  325 0)
 (548 equiv "( ( move 3 , ua ; pen ul ) ; move ua , 2 ) " ((34 del 5)) () 34
  0)
 (549 uniq "( move ua , ul ; pen 4 ; move 4 / 2 , ul ) "
  ((45 del 5) (67 del 5) (484 sub 3) (540 sub 3)) () -1 0)
 (550 uniq "( pen 4 - ul ; move ul , 4 ; move ua , 4 ) " () () -1 0)
 (551 uniq "( pen 2 ; move ul , ul ; move 4 , ul / 3 ) " () () -1 0)
 (552 equiv "( move ua , ua ; move 0 , 3 ; move ua , ul ) "
  ((454 sub 3) (494 sub 3) (505 sub 3) (559 sub 3) (564 sub 3) (567 sub 3)
   (629 sub 1) (1224 ins 4))
  () 33 0)
 (553 equiv "( ( move 4 , 3 ; move ua , ul ) ; pen ul ) " () () 169 0)
 (554 equiv "( pen ua ; ( move 4 , ul ; v0 = ua ) ) " () () 16 0)
 (555 equiv "( move ua , ul ; move 0 , ua ; move 4 , ua ) "
  ((65 del 5) (558 sub 3) (559 sub 2) (563 sub 3) (626 sub 3)) () 1 0)
 (556 uniq "( pen 4 ; ( move ua , ul ; move 4 , ul ) ) "
  ((67 del 5) (577 sub 3) (623 sub 3) (636 sub 3)) () -1 0)
 (557 equiv "( v0 = 3 + 2 ; pen 4 ; move v0 , 3 ) " () () 145 0)
 (558 equiv "( move ua , ul ; move 2 , ul ; move 2 , ua ) "
  ((555 sub 3) (559 sub 3) (563 sub 2) (626 sub 3) (824 ins 2)) () 67 0)
 (559 equiv "( move ua , ul ; move 0 , ul ; move 4 , ul ) "
  ((35 del 5) (67 del 5) (500 sub 3) (552 sub 3) (555 sub 2) (558 sub 3)
   (563 sub 1) (567 sub 2) (632 sub 3))
  () 38 0)
 (560 equiv "( pen ua * 3 ; move 2 + 2 , ul ; pen ul ) " ((47 del 5)) () 16
  0)
 (561 equiv "( pen 3 + ul ; ( pen ua ; move ua , 0 ) ) " () () 14 0)
 (562 equiv "( move ua , 2 ; move ua , ul ; v0 = ua ) "
  ((62 del 5) (508 sub 3) (566 sub 3) (571 sub 3) (574 sub 3) (1238 ins 4))
  () 62 0)
 (563 uniq "( move ua , ul ; move 2 , ul ; move 4 , ul ) "
  ((67 del 5) (454 sub 3) (555 sub 3) (558 sub 2) (559 sub 1) (567 sub 2)
   (632 sub 3))
  () -1 0)
 (564 uniq "( move ul , ua ; move 3 , 4 ; move ua , ul ) "
  ((456 sub 3) (498 sub 3) (552 sub 3) (629 sub 3)) () -1 0)
 (565 equiv "( move ua , ua ; v0 = ul ; move ua , 4 ) "
  ((33 del 5) (63 del 5) (64 del 5) (505 sub 3) (610 sub 3)) () 33 0)
 (566 equiv "( move ua , ua ; move ua , 4 ; v0 = ul ) "
  ((63 del 5) (502 sub 2) (506 sub 3) (508 sub 2) (562 sub 3) (571 sub 2)
   (574 sub 3) (647 sub 3) (1195 ins 4) (1397 ins 6))
  () 33 0)
 (567 uniq "( move ua , ua ; move 4 , ul ; move 4 , ul ) "
  ((45 del 5) (66 del 5) (67 del 5) (552 sub 3) (559 sub 2) (563 sub 2)
   (571 sub 3) (629 sub 3))
  () -1 0)
 (568 equiv "( pen ul ; pen 4 * 3 ; move ua , ua * 3 ) " () () 14 0)
 (569 equiv "( pen 3 ; ( pen ua ; move ul - ua , 2 ) ) " ((72 del 5)) () 72
  0)
 (570 equiv "( pen ua ; move 2 * 2 , ul ; pen ul / 4 ) " () () 16 0)
 (571 equiv "( move ua , ua ; move 4 , ul ; v0 = ul ) "
  ((66 del 5) (502 sub 3) (506 sub 2) (562 sub 3) (566 sub 2) (567 sub 3)
   (574 sub 1) (620 sub 3))
  () 35 0)
 (572 equiv "( pen ua ; move ul , ua + 2 ; move ua , 0 ) " () () 70 0)
 (573 equiv "( ( pen ul ; pen 2 + ul ) ; move ua , 4 ) " ((48 del 5)) () 13
  0)
 (574 equiv "( move ua , ul ; move 4 , ul ; v0 = ul ) "
  ((67 del 5) (506 sub 1) (508 sub 3) (562 sub 3) (566 sub 3) (571 sub 1)) ()
  67 0)
 (575 uniq "( move 3 + 2 , 0 - 3 ; move ua + 2 , ul ) " () () -1 0)
 (576 equiv "( pen 3 ; move ua , ua - ul ; move 0 , ua ) "
  ((83 del 5) (650 sub 3)) () 83 0)
 (577 equiv "( pen 3 ; ( pen ua - ul ; move 4 , ul ) ) " ((556 sub 3)) () 16
  0)
 (578 equiv "( pen ua ; ( move 4 , ua * 3 ; pen ul ) ) " ((111 del 4)) () 12
  0)
 (579 equiv "( pen ua + 2 ; move ua , 2 ; move ua , ul ) "
  ((62 del 5) (517 sub 1) (586 sub 3) (644 sub 3) (646 sub 2) (648 sub 2)
   (673 sub 3))
  () 517 0)
 (580 equiv "( pen ul ; move ul , ul ; move ua + 3 , 0 ) " () () 1 0)
 (581 equiv "( pen ua + ul ; move 3 , ua ; move ua , 2 ) "
  ((34 del 5) (461 sub 3) (526 sub 2) (583 sub 3) (593 sub 3)) () 526 0)
 (582 uniq "( move ua + ul , ua ; pen 3 ; move ua , 2 ) "
  ((61 del 5) (191 del 3)) () -1 0)
 (583 uniq "( pen ul + ul ; move 3 , 0 ; move ua , ul ) "
  ((461 sub 3) (581 sub 3) (593 sub 3)) () -1 0)
 (584 equiv "( move ua , ua ; move ua , 2 * 2 ; pen ua ) "
  ((61 del 5) (635 sub 2)) () 33 0)
 (585 equiv "( move 4 , ua + ul ; pen ua ; move ul , 0 ) " ((470 sub 3)) ()
  325 0)
 (586 equiv "( pen ua + 3 ; move ul , 2 ; move ua , ua ) "
  ((491 sub 3) (579 sub 3) (626 sub 3) (648 sub 3)) () 70 0)
 (587 uniq "( pen 3 ; move ua , 0 - ua ; move ul , ul ) " () ((623 nul 4)) -1
  0)
 (588 equiv "( move 3 , 0 ; pen ul ; move ua , ul + ul ) " () () 34 0)
 (589 uniq "( move 0 - ua , ua + 3 ; move 4 , 3 + 3 ) " () () -1 0)
 (590 uniq "( move ua , 3 ; pen ua + ul ; move ul , 2 ) "
  ((529 sub 3) (591 sub 3)) () -1 0)
 (591 equiv "( move 4 , ul ; pen ua + ul ; move ul , 0 ) "
  ((590 sub 3) (606 sub 2) (621 sub 3) (892 ins 2)) () 325 0)
 (592 uniq "( move ua , 0 ; pen ua ; move ul - 4 , ul ) "
  ((35 del 5) (66 del 5) (531 sub 3) (595 sub 3) (639 sub 3) (659 sub 1))
  ((659 sub 1)) -1 0)
 (593 uniq "( pen 3 + ul ; move 2 , ua ; move ua , ul ) "
  ((82 del 5) (461 sub 3) (517 sub 3) (581 sub 3) (583 sub 3)) () -1 0)
 (594 equiv "( move ua , 0 ; move 4 , ul - ua ; pen ul ) " ((35 del 5)) () 35
  0)
 (595 uniq "( move 4 , 0 ; pen ua ; move ul + ul , ul ) "
  ((515 sub 3) (592 sub 3)) () -1 0)
 (596 uniq "( pen ua ; move ua , ul ; move 2 , 4 + ul ) " ((674 sub 3)) () -1
  0)
 (597 uniq "( pen ua ; move ua , 4 ; move ua + ul , 3 ) " ((645 sub 3)) () -1
  0)
 (598 uniq "( move ua , ua ; pen 4 ; move ul + ul , 2 ) " ((640 sub 3)) () -1
  0)
 (599 equiv "( move ua , ua ; pen 2 ; move ua , 4 - ul ) "
  ((62 del 5) (63 del 5) (522 sub 3)) () 281 0)
 (600 uniq "( pen ua ; move 4 + ul , 2 ; move ua , ul ) "
  ((40 del 5) (612 sub 3) (622 sub 3)) () -1 0)
 (601 uniq "( pen ua ; move 3 , ua - 4 ; move ua , ua ) "
  ((479 sub 3) (650 sub 3)) () -1 0)
 (602 equiv "( pen ul / ua ; move ul , ua ; move 4 , 0 ) " ((71 del 5)) () 0
  0)
 (603 equiv "( pen ua ; move ul - ua , 2 ; v0 = ua ) " ((72 del 5)) () 72 0)
 (604 uniq "( move ua , ul / 2 ; pen ua ; move 3 , ua ) " ((607 sub 3)) () -1
  0)
 (605 uniq "( move 2 - ua , ul ; move ua , ul ; pen 4 ) " ((64 del 5))
  ((935 nul 3) (1225 nul 7)) -1 0)
 (606 equiv "( move 4 , ul ; pen ua + 3 ; move ul , ua ) "
  ((325 del 2) (591 sub 2)) () 325 0)
 (607 uniq "( move ua , 4 * 2 ; pen ua ; move ul , ua ) "
  ((604 sub 3) (663 sub 3)) () -1 0)
 (608 uniq "( move ua + 3 , ul / 3 ; move 3 + 2 , 2 ) " () () -1 0)
 (609 equiv "( pen ul / ua ; move ua , 0 ; v0 = ul ) "
  ((71 del 5) (97 del 5) (664 sub 1)) () 0 0)
 (610 uniq "( move ua , ua ; pen 4 - ul ; move ua , 4 ) "
  ((41 del 5) (63 del 5) (64 del 5) (527 sub 3) (565 sub 3)) () -1 0)
 (611 uniq "( pen ua * 3 ; move ua , ul ; move ua , 4 ) "
  ((64 del 5) (79 del 5) (526 sub 3)) () -1 0)
 (612 uniq "( pen ua ; move 4 , ul - 3 ; move ua , ul ) "
  ((600 sub 3) (670 sub 3)) () -1 0)
 (613 uniq "( move ul - 4 , ua ; pen ua ; move 3 , ul ) "
  ((427 sub 3) (523 sub 3)) () -1 0)
 (614 uniq "( pen 4 ; move ua , ua ; move 4 - ua , ul ) " ((66 del 5)) () -1
  0)
 (615 uniq "( pen ua ; move 4 , 3 ; move ul - ua , ul ) " () () -1 0)
 (616 uniq "( move 3 , ul ; pen 4 ; move ua , ua - ul ) "
  ((522 sub 3) (675 sub 3)) () -1 0)
 (617 uniq "( pen ua ; move 4 , ul ; move ua - ul , 4 ) "
  ((44 del 5) (86 del 5)) () -1 0)
 (618 uniq "( move 4 / 2 , 3 ; move ua + 3 , 2 - ul ) " () ((1170 nul 10)) -1
  0)
 (619 uniq "( pen ua ; move ua , ul ; move 3 , ul / 4 ) " () () -1 0)
 (620 equiv "( pen ua * ua ; move 4 , ua ; v0 = ul ) " ((571 sub 3)) () 12 0)
 (621 uniq "( move 4 , ul ; pen ul + ul ; move 4 , ul ) "
  ((535 sub 3) (591 sub 3)) () -1 0)
 (622 uniq "( pen 4 ; move 4 - ul , ul ; move ua , ul ) " ((600 sub 3)) () -1
  0)
 (623 equiv "( pen 3 ; ( move ua , ua ; move ul , ul ) ) " ((556 sub 3)) ()
  587 0)
 (624 equiv "( ( pen ua ; move ul , ul ) ; move 4 , ul ) "
  ((328 del 2) (496 sub 3)) () 328 0)
 (625 equiv "( pen ua ; move ul - ua , 2 ; pen ua * 2 ) "
  ((72 del 5) (74 del 5)) () 72 0)
 (626 equiv "( move ua , ul ; move ul , 2 ; move ua , ua ) "
  ((491 sub 2) (494 sub 3) (555 sub 3) (558 sub 3) (586 sub 3) (629 sub 3))
  () 193 0)
 (627 equiv "( pen 4 + ul ; move ua + 2 , ua ; pen ul ) " ((80 del 5)) () 69
  0)
 (628 equiv "( pen ua - ul ; pen 3 ; move ua + 3 , ul ) " () () 290 0)
 (629 uniq "( move ua , ua ; move ul , 3 ; move ua , ul ) "
  ((454 sub 3) (491 sub 3) (552 sub 1) (564 sub 3) (567 sub 3) (626 sub 3)
   (1485 ins 6))
  () -1 0)
 (630 equiv "( pen 2 / ua ; move 3 + ul , ua ; pen ua ) " () () 0 0)
 (631 equiv "( pen 2 - ul ; pen ua ; move 4 , ul - ua ) " () () 16 0)
 (632 uniq "( move ul , ul ; move ua , ul ; move 3 , ul ) "
  ((497 sub 3) (559 sub 3) (563 sub 3)) () -1 0)
 (633 uniq "( pen ua - ul ; move 4 * 2 , ul ; pen ua ) " ((88 del 5)) () -1
  0)
 (634 equiv "( pen ua ; move ua + ul , ua ; move ul , 0 ) " () () 69 0)
 (635 equiv "( move ua , ua ; move ua , ua + 2 ; pen ua ) "
  ((61 del 5) (510 sub 2) (584 sub 2)) () 32 0)
 (636 equiv "( pen 4 ; ( move ua , ul ; pen ua - ul ) ) " ((556 sub 3)) () 17
  0)
 (637 equiv "( v0 = 4 ; move ua , ul ; move v0 , 0 ) " ((1328 ins 5)) () 1 0)
 (638 equiv "( move 3 , ua ; v0 = 2 ; move ua , v0 ) " ((669 sub 1)) () 34 0)
 (639 uniq "( move ua , 0 ; pen ua ; move ul , ua - ul ) "
  ((464 sub 3) (592 sub 3) (675 sub 1)) () -1 0)
 (640 equiv "( move ul , ua ; pen ua ; move ua + ul , 2 ) "
  ((70 del 5) (598 sub 3)) () 70 0)
 (641 uniq "( move 3 , 4 + 2 ; move ua + ul , ua + 2 ) " ((677 sub 3)) () -1
  0)
 (642 equiv "( pen ua ; move ua , ua + ua ; move ul , 4 ) "
  ((95 del 5) (320 del 2) (650 sub 3)) () 320 0)
 (643 uniq "( move ul , ul + ul ; pen ua ; move ua , 2 ) " () () -1 0)
 (644 uniq "( pen ua + ua ; move ua , ul ; move 3 , ul ) "
  ((525 sub 3) (579 sub 3) (646 sub 3) (648 sub 3) (673 sub 2)) () -1 0)
 (645 uniq "( pen ua ; move ua , ua ; move 3 + ul , ul ) "
  ((78 del 5) (201 del 3) (597 sub 3)) ((868 nul 4)) -1 0)
 (646 uniq "( pen ua + ul ; move ua , 2 ; move ul , ul ) "
  ((517 sub 3) (579 sub 2) (644 sub 3) (673 sub 1)) () -1 0)
 (647 equiv "( move ua , ua ; move ua , 4 ; pen ua + ul ) "
  ((63 del 5) (566 sub 3)) () 33 0)
 (648 equiv "( pen ua + 2 ; move ul , ul ; move ua , ul ) "
  ((358 del 2) (517 sub 3) (579 sub 2) (586 sub 3) (644 sub 3) (673 sub 3))
  () 358 0)
 (649 uniq "( pen ul ; move ua , ua ; move ua , ul - 3 ) "
  ((76 del 5) (520 sub 2) (652 sub 3)) () -1 0)
 (650 uniq "( pen ua ; move ua , ua - ul ; move ua , 4 ) "
  ((63 del 5) (64 del 5) (86 del 5) (479 sub 3) (576 sub 3) (601 sub 3)
   (642 sub 3) (670 sub 3))
  () -1 0)
 (651 uniq "( move ua , ua ; pen ua * 2 ; move ul , ul ) " ((474 sub 3))
  ((889 nul 5)) -1 0)
 (652 equiv "( pen ul ; move 3 , ua ; move ua , ul + ul ) "
  ((520 sub 3) (532 sub 3) (649 sub 3) (658 sub 3) (674 sub 3)) () 34 0)
 (653 equiv "( move ua + 3 , ul ; move ul , ua ; pen ul ) " ((209 del 3)) ()
  1 0)
 (654 uniq "( move ul - ua , ua ; move 3 + 2 , 2 - 4 ) " () () -1 0)
 (655 equiv "( move ua , ul ; v0 = 4 ; move v0 , 4 ) "
  ((668 sub 3) (671 sub 1)) () 43 0)
 (656 equiv "( move ua , 4 ; v0 = ul ; move 4 , v0 ) " ((669 sub 3)) () 45 0)
 (657 equiv "( move ua , ul - ua ; pen ul ; move ua , 4 ) "
  ((63 del 5) (64 del 5) (204 del 3)) () 64 0)
 (658 uniq "( pen ua ; move ul , ua ; move 4 , ul + ul ) "
  ((652 sub 3) (674 sub 2)) () -1 0)
 (659 equiv "( move ua , ua ; pen ua ; move ul - 4 , ul ) "
  ((66 del 5) (592 sub 1)) () 592 0)
 (660 uniq "( move 2 , ul ; pen ua ; move ua , ul / ua ) " () () -1 0)
 (661 uniq "( move ua + ul , 3 + ul ; move 4 , 4 + 2 ) "
  ((38 del 6) (43 del 6)) () -1 0)
 (662 uniq "( move ua , 2 - 4 ; move 4 - ul , 2 - ua ) " () () -1 0)
 (663 uniq "( move ua , 4 / ua ; pen ua ; move ul , ul ) " ((607 sub 3)) ()
  -1 0)
 (664 equiv "( pen ul / ua ; move ua , ul ; v0 = ul ) "
  ((71 del 5) (97 del 5) (609 sub 1) (983 ins 2)) () 0 0)
 (665 equiv "( pen ua + ul ; pen ua ; move ua - ul , 4 ) "
  ((86 del 5) (667 sub 3)) () 85 0)
 (666 equiv "( v0 = 4 ; pen 4 + ul ; move v0 , ul ) " () () 16 0)
 (667 equiv "( pen ul + ul ; pen ua ; move ul - ua , 4 ) "
  ((85 del 5) (665 sub 3)) () 85 0)
 (668 equiv "( move ua , ua ; v0 = ua ; move v0 , 2 ) "
  ((655 sub 3) (671 sub 3)) () 32 0)
 (669 equiv "( move ua , ua ; v0 = 2 ; move ua , v0 ) "
  ((638 sub 1) (656 sub 3)) () 32 0)
 (670 equiv "( pen ua ; move ua , ul + ul ; move ua , ul ) "
  ((240 del 3) (612 sub 3) (650 sub 3)) () 517 0)
 (671 equiv "( move ua , ul ; v0 = 4 ; move v0 , ul ) "
  ((655 sub 1) (668 sub 3)) () 67 0)
 (672 uniq "( move 0 - ua , ul / ua ; move 4 , ua + 2 ) "
  ((38 del 6) (65 del 6)) () -1 0)
 (673 equiv "( pen ua + ul ; move ua , ul ; move ul , ul ) "
  ((359 del 2) (579 sub 3) (644 sub 2) (646 sub 1) (648 sub 3)) () 359 0)
 (674 uniq "( pen ua ; move ul , ul ; move ua , ul + ul ) "
  ((358 del 2) (596 sub 3) (652 sub 3) (658 sub 2)) () -1 0)
 (675 uniq "( move ua , ul ; pen ua ; move ul , ua - ul ) "
  ((351 del 2) (616 sub 3) (639 sub 1)) () -1 0)
 (676 uniq "( move ua / 2 , 3 ; move ua + ul , 3 + ul ) " ((36 del 6)) () -1
  0)
 (677 uniq "( move ul , 4 + 2 ; move ua + ul , 4 + ul ) " ((641 sub 3)) () -1
  0)
 (678 uniq "( move 2 * 2 , ua * 2 ; move ul - ua , ul ) " ((120 del 4)) () -1
  0)
 (679 uniq "( move 3 + ul , ua ; move ul - ua , 2 / 3 ) "
  ((34 del 6) (158 del 4)) () -1 0)
 (680 uniq "( move 4 + ul , ul + ul ; move 3 * 2 , ul ) " () () -1 0)
 (681 uniq "( move ua , ul / 4 ; move ua - ul , 3 - 2 ) " () () -1 0)
 (682 uniq "( move ua + 3 , 3 / 4 ; move ul + ul , ul ) " () () -1 0)
 (683 equiv "( pen ua ; v0 = ua ; move 3 + ul , v0 ) " () () 12 0)
 (684 equiv "( pen 3 + ul ; v0 = ul ; move ua , v0 ) " ((708 sub 3)) () 17 0)
 (685 equiv "( v0 = ua ; pen 4 + ul ; move ua , v0 ) " () () 14 0)
 (686 uniq "( move 4 , ua ; ( move 2 , 2 ; move 3 , 0 ) ) " () () -1 0)
 (687 uniq "( move 4 , 3 ; move 4 , ul ; move 3 + 3 , 3 ) " () () -1 0)
 (688 uniq "( move 2 , 2 / 3 ; pen ua - 2 ; move 2 , 2 ) " () () -1 0)
 (689 uniq "( v0 = 4 * 2 ; pen 4 ; move v0 , v0 ) " () () -1 0)
 (690 uniq "( move 3 , 4 ; pen 4 + ul ; move 3 , 2 - 3 ) " () () -1 0)
 (691 uniq "( move 2 , 2 ; ( pen ua ; move 4 , 3 + 2 ) ) " () () -1 0)
 (692 uniq "( pen 4 ; move 3 , 4 ; move ua - 2 , 3 / 2 ) " () () -1 0)
 (693 uniq "( move 4 + 2 , ua * ua ; move ua + ul , ul ) " () () -1 0)
 (694 uniq "( move 4 , ua + ua ; move ua + ul , ua / 3 ) " () () -1 0)
 (695 equiv "( move 3 + ul , ua * 3 ; move ua , ul + ul ) " ((699 sub 2)) ()
  20 0)
 (696 uniq "( pen 4 * 3 ; move 4 , 4 ; move ul , 2 - 3 ) " () () -1 0)
 (697 uniq "( ( move 3 , 4 * 4 ; pen ua ) ; move 3 , 0 ) " () () -1 0)
 (698 equiv "( v0 = 2 ; move 4 , v0 ; move ua , v0 ) " () () 253 0)
 (699 uniq "( move 3 + ul , ul / 3 ; move ua , ul + ul ) " ((695 sub 2)) ()
  -1 0)
 (700 equiv "( move 4 , ua * 3 ; move ua , 2 ; move 0 , 2 ) " () () 20 0)
 (701 uniq "( move 0 , ua - 3 ; move ua , 2 ; move 4 , 4 ) " () () -1 0)
 (702 uniq "( move 2 , 4 + 4 ; move ua , 0 ; move ua , 4 ) " () () -1 0)
 (703 uniq "( ( pen 3 ; move 4 , 4 + ul ) ; move 4 , 4 ) " () () -1 0)
 (704 uniq "( move 4 , ua ; move 3 , 3 + 2 ; move 3 , ua ) " () () -1 0)
 (705 equiv "( move 4 + 4 , 0 ; v0 = ul ; move ua , 2 ) " () () 124 0)
 (706 equiv "( pen ua ; v0 = ul ; move ua * v0 , ua ) " () () 14 0)
 (707 uniq "( move ul , 2 ; move ua , 4 * 2 ; move 3 , 4 ) " () () -1 0)
 (708 equiv "( pen ul / ua ; v0 = ul ; move ua , v0 ) "
  ((71 del 6) (684 sub 3)) () 0 0)
 (709 equiv "( move ua , 3 / 4 ; move 4 , ua ; v0 = 0 ) " () () 379 0)
 (710 uniq "( move 3 + 2 , 4 ; move ua , 4 ; v0 = ul ) " ((763 sub 3)) () -1
  0)
 (711 uniq "( move ua , 4 ; move 2 - 4 , ul ; move 4 , 4 ) " ((761 sub 1)) ()
  -1 0)
 (712 equiv "( move 4 , 2 ; ( move ua , ul ; v0 = 0 ) ) " () () 40 0)
 (713 uniq "( move 2 , 4 ; ( move ua , 2 ; move ua , 4 ) ) " () () -1 0)
 (714 equiv "( v0 = ua ; pen 3 + 3 ; move v0 , v0 ) " () () 14 0)
 (715 uniq "( pen 2 ; move ua , 3 + 2 ; move 2 + ul , 2 ) " () () -1 0)
 (716 equiv "( move 3 - ua , 0 ; pen ua * 3 ; move 4 , 0 ) " () () 12 0)
 (717 uniq "( pen 3 + ul ; move 2 , 0 ; move 4 + ul , 2 ) " () () -1 0)
 (718 uniq "( move 4 , ua / 4 ; move 3 , ul ; move 3 , 4 ) " () () -1 0)
 (719 uniq "( pen 3 + ul ; move ua , 3 * 3 ; move 3 , 0 ) " () () -1 0)
 (720 uniq "( pen 3 ; move 3 , 2 - ua ; move 3 + 2 , ul ) " () () -1 0)
 (721 uniq "( move ul , 3 ; pen ua ; move 4 + 3 , 3 + 2 ) " () () -1 0)
 (722 uniq "( move 3 , 4 + 3 ; pen ua * 3 ; move ul , 0 ) " () () -1 0)
 (723 uniq "( move 2 , 3 + 2 ; pen ua + 4 ; move 4 , ul ) " () () -1 0)
 (724 uniq "( pen ul ; move 3 * 2 , 3 * 2 ; move ua , 2 ) " () () -1 0)
 (725 equiv "( pen 4 ; move ua , 4 + 2 ; v0 = 0 - ul ) " () () 25 0)
 (726 uniq "( move 3 * 2 , 4 ; pen ua ; move ua , 2 * 2 ) " () () -1 0)
 (727 uniq "( move 2 , ua - ul ; pen 3 ; move 3 , 2 / 3 ) " () () -1 0)
 (728 equiv "( v0 = ua ; ( pen v0 ; move 4 , v0 ) ) " () () 12 0)
 (729 uniq "( move ua , 0 ; move ua , 2 ; move 2 , 0 - ul ) "
  ((733 sub 3) (829 sub 3)) () -1 0)
 (730 uniq "( pen 4 ; move 4 * 2 , 3 ; move ul , ua + 3 ) " () () -1 0)
 (731 uniq "( ( pen 3 ; move 2 + 2 , 3 ) ; move ua , ul ) " () () -1 0)
 (732 uniq "( pen ua - 4 ; move 3 , 4 ; move 3 + ul , 4 ) " () () -1 0)
 (733 equiv "( move ua , 0 ; move ua , 2 ; move 0 , 4 + ul ) "
  ((729 sub 3) (918 sub 3)) () 32 0)
 (734 uniq "( move 3 , 4 ; pen 3 ; move 2 + ul , 4 / ua ) " () () -1 0)
 (735 equiv "( v0 = ua ; move v0 , ua ; move v0 , 2 ) " () () 32 0)
 (736 uniq "( move 3 , 0 - ul ; move ul , 0 ; move ua , 2 ) " () () -1 0)
 (737 uniq "( move 3 , 3 ; pen 2 * 2 ; move ul , ul / 4 ) " () () -1 0)
 (738 uniq "( move ul - ua , 2 / ua ; move ua , ua - ul ) " ((62 del 6)) ()
  -1 0)
 (739 equiv "( v0 = ua - ul ; move v0 + v0 , v0 ) " ((1451 ins 5)) () 2 0)
 (740 uniq "( move ua , ul / ua ; move ua - ul , ua + 4 ) "
  ((63 del 6) (64 del 6) (194 del 4) (197 del 4)) () -1 0)
 (741 uniq "( move ua , 0 ; move ua , ua + 2 ; move 4 , 4 ) "
  ((148 del 5) (755 sub 3)) ((1183 nul 5)) -1 0)
 (742 equiv "( move ua , 0 ; move 4 , ua + ul ; v0 = 0 ) " ((155 del 5)) ()
  35 0)
 (743 equiv "( move ua , ul ; v0 = 4 ; move v0 , v0 ) " ((1321 ins 4)) () 43
  0)
 (744 equiv "( v0 = ua ; move v0 , ul ; move 4 , v0 ) " () () 1 0)
 (745 uniq "( move ua , 3 ; move 3 - ul , 0 ; move ua , 3 ) " ()
  ((1098 nul 10)) -1 0)
 (746 equiv "( v0 = 4 ; move ua , v0 ; move v0 , ul ) " ((1384 ins 4)) () 45
  0)
 (747 uniq "( pen 3 ; ( move 4 , 2 / 3 ; move ua , ul ) ) " () () -1 0)
 (748 uniq "( move 3 , 0 ; move ul , 4 ; move ua , 2 + ul ) " () () -1 0)
 (749 equiv "( move ua , ul ; move 2 , 2 ; move 2 , ua * 3 ) " () () 38 0)
 (750 uniq "( move ua , 0 ; move 4 , ul - 2 ; move ua , 4 ) " () () -1 0)
 (751 uniq "( move ua , 3 ; move ua , 0 - 4 ; move ul , 4 ) " () () -1 0)
 (752 uniq "( move ul , 0 ; move ua , 3 - 4 ; move ua , 4 ) " () () -1 0)
 (753 uniq "( move ua , 0 ; ( move 3 , 2 ; move ua , ua ) ) " () () -1 0)
 (754 uniq "( move 3 , 2 ; move 3 , 3 - ua ; move ua , ul ) " () () -1 0)
 (755 uniq "( move 4 , 0 ; move ul , ua + ul ; move 4 , 4 ) " ((741 sub 3))
  () -1 0)
 (756 uniq "( move 3 , ua ; move 2 , ul ; move 4 + ul , 4 ) " () () -1 0)
 (757 uniq "( move ua , 3 - 4 ; move 3 , ul ; move 2 , ul ) " () () -1 0)
 (758 uniq "( move ua , ua ; move 4 , ua - 4 ; move 4 , 3 ) " () () -1 0)
 (759 uniq "( move ua , 2 ; move ua + ul , 4 ; v0 = 4 ) " () () -1 0)
 (760 uniq "( move 3 , 2 + ul ; move 4 , ul ; move 4 , ul ) " () () -1 0)
 (761 uniq "( move ua , ul ; move 2 - 4 , ul ; move 4 , 4 ) "
  ((132 del 5) (711 sub 1)) () -1 0)
 (762 equiv "( move ua + 4 , ul ; move 0 , ua + 3 ; pen 0 ) " ((151 del 5))
  () 1 0)
 (763 equiv "( move 2 * 2 , ul ; move ua , 4 ; v0 = ul ) " ((710 sub 3)) ()
  44 0)
 (764 equiv "( pen 2 + 2 ; ( move 4 + ul , 3 ; pen ul ) ) " () () 145 0)
 (765 equiv "( move ua , 4 ; move ua , 4 ; v0 = 2 / ua ) " () () 41 0)
 (766 uniq "( ( move 4 , ul ; move 2 , ul ) ; move ua , 2 ) " () () -1 0)
 (767 uniq "( pen ua ; move 3 , ua + 2 ; move ua + 3 , 2 ) " () () -1 0)
 (768 uniq "( pen ua + 2 ; move 4 , ua ; move ua , 2 + 2 ) " ((774 sub 3))
  ((1448 nul 13)) -1 0)
 (769 equiv "( move ua , 4 ; move 4 , ul ; v0 = ua * 3 ) " () () 45 0)
 (770 uniq "( pen ua - 3 ; move ua , 0 ; move 2 + ul , 4 ) " () () -1 0)
 (771 uniq "( move 4 , ua ; pen ua * 2 ; move ua , 0 - 2 ) " ((286 del 4)) ()
  -1 0)
 (772 equiv "( pen ua ; v0 = 2 ; move v0 * v0 , ul ) " () () 16 0)
 (773 equiv "( v0 = ua + 4 ; pen v0 ; move v0 , ul ) " () () 77 0)
 (774 uniq "( pen ua + 3 ; move ul , 4 ; move ua , 2 + 2 ) " ((768 sub 3)) ()
  -1 0)
 (775 equiv "( pen ua * 4 ; v0 = ua ; move v0 , v0 ) " () () 14 0)
 (776 uniq "( pen ua + 2 ; move 2 + ul , ul ; move 4 , 2 ) " () () -1 0)
 (777 equiv "( move ua , 2 - ua ; move 4 , 4 - ua ; pen 0 ) " () () 21 0)
 (778 uniq "( pen 3 + 2 ; move ua , ul ; move 3 , ua - 2 ) " ((290 del 4)) ()
  -1 0)
 (779 equiv "( pen 4 + ul ; move 4 , 0 - ul ; move 0 , ul ) " () () 27 0)
 (780 equiv "( move ua , 4 ; ( move ua , 4 ; v0 = ua ) ) " ((843 sub 1)) ()
  41 0)
 (781 uniq "( move ua , 3 + 3 ; pen 2 + ul ; move ua , 3 ) " () () -1 0)
 (782 uniq "( pen ua + ul ; move 2 , 2 - ul ; move 4 , 2 ) " () () -1 0)
 (783 equiv "( pen ua + 3 ; move ua + 4 , ul ; v0 = 0 ) " ((121 del 5)) () 77
  0)
 (784 uniq "( move ua + 3 , ul ; pen ua + 2 ; move 4 , 3 ) " ((294 del 4)) ()
  -1 0)
 (785 equiv "( v0 = ua ; pen v0 * v0 ; move 4 , ul ) " () () 16 0)
 (786 uniq "( move 4 + ul , 0 ; pen 2 ; move 4 , ul - ua ) " () () -1 0)
 (787 uniq "( pen ua ; move 4 + 2 , ul ; move 3 + ul , 3 ) " () () -1 0)
 (788 uniq "( pen 4 * 3 ; move ua , ul ; move ua + 2 , 2 ) " () () -1 0)
 (789 uniq "( pen 3 + ul ; move 4 , 4 ; move ua - ul , 0 ) " () () -1 0)
 (790 equiv "( pen ua ; v0 = 3 - ua ; move 4 , 0 - ul ) " () () 27 0)
 (791 uniq "( move ul , 4 + 4 ; pen ua ; move ua + 4 , 2 ) " ((289 del 4)) ()
  -1 0)
 (792 uniq "( pen 4 ; move 4 , ua + ua ; move 2 + ul , 4 ) " () () -1 0)
 (793 uniq "( pen 4 ; move ua + 4 , 0 - 4 ; move ul , ul ) " () () -1 0)
 (794 uniq "( pen 3 ; move 3 , 2 ; move ua - ul , 4 + ul ) " () () -1 0)
 (795 uniq "( pen ua ; move ul , 3 - 2 ; move 4 , ul - 3 ) " () () -1 0)
 (796 uniq "( move ul - ua , 0 ; pen 4 ; move 3 / ua , 4 ) " () () -1 0)
 (797 uniq "( ( pen 2 ; move ua , 0 ) ; move 4 + ul , ul ) " () () -1 0)
 (798 uniq "( move ul + ul , 4 ; pen 3 ; move ul - 4 , 2 ) " () () -1 0)
 (799 uniq "( move ua , 4 + ul ; move ua , 3 ; pen 4 / 2 ) " ()
  ((1455 nul 10)) -1 0)
 (800 equiv "( move ua , ua ; move 0 , 3 + ul ; move ua , 0 ) " () () 33 0)
 (801 uniq "( move ua , 2 ; pen 4 + ul ; move 4 / 3 , ua ) " () () -1 0)
 (802 uniq "( move 4 + ul , 4 ; pen ua * 3 ; move 2 , ul ) " () () -1 0)
 (803 uniq "( move 2 , 3 ; ( pen ua ; move ua , ul - 2 ) ) " () () -1 0)
 (804 equiv "( move ua , ul - ua ; move ua + ul , ul + ul ) "
  ((236 del 4) (487 del 2)) () 193 0)
 (805 equiv "( move ua , ul ; ( move 3 + ul , 2 ; pen 3 ) ) " () () 38 0)
 (806 equiv "( move 0 , 0 ; move ua , ua ; move 3 + ul , ul ) " ((201 del 5))
  () 35 0)
 (807 uniq "( pen 3 ; move ul + ul , ua * 4 ; move 4 , 4 ) " () () -1 0)
 (808 uniq "( move ul , 3 ; pen ua / 2 ; move ua , 3 / 4 ) " () () -1 0)
 (809 uniq "( ( move 3 - ul , ul ; pen ua ) ; move 3 , 2 ) " () () -1 0)
 (810 equiv "( move 0 , ul - ua ; move 4 , ua ; move ua , 0 ) " () () 1 0)
 (811 equiv "( ( pen ua ; move 3 , 4 ) ; move 3 , ua + ul ) " ((530 del 2))
  () 530 0)
 (812 uniq "( ( move 4 + ul , 4 ; pen ua ) ; move ua , 2 ) "
  ((125 del 5) (289 del 4)) () -1 0)
 (813 uniq "( move 4 , 3 - ul ; pen ua * 4 ; move ua , 4 ) " () () -1 0)
 (814 equiv "( move 0 , ul ; move ua + 3 , 0 ; move ul , ul ) " () () 209 0)
 (815 uniq "( ( pen 4 ; move 3 , ua ) ; move 2 + ul , ul ) " () () -1 0)
 (816 uniq "( pen 4 ; ( move 2 , 3 ; move ua , ul + ul ) ) " () () -1 0)
 (817 uniq "( move 3 , ul ; ( pen ua * 4 ; move ua , 2 ) ) " () () -1 0)
 (818 uniq "( move ua - ul , 0 ; move ua , 2 ; move 3 , ul ) " () () -1 0)
 (819 uniq "( move 2 , 2 ; move ua , ua + ua ; move ua , 4 ) " () () -1 0)
 (820 uniq "( move ua , 2 - 3 ; move 0 , ul ; move ua , ul ) " () () -1 0)
 (821 equiv "( pen ua + ul ; move 4 , ul - 2 ; pen 0 - 3 ) " () () 27 0)
 (822 uniq "( move ua , ua - 4 ; move 0 , ua ; move ua , 4 ) " ((828 sub 3))
  () -1 0)
 (823 equiv "( move ua + 4 , ul ; move ul , 3 ; move 0 , ua ) " ((151 del 5))
  () 1 0)
 (824 equiv "( move ua , ul ; move 2 , 3 + ul ; move 2 , ua ) " ((558 del 2))
  () 43 0)
 (825 uniq "( move 3 , ul ; move ul , ua + 2 ; move ua , 2 ) " () () -1 0)
 (826 equiv "( ( pen 4 ; v0 = ua + 4 ) ; move ua , ul ) " () () 17 0)
 (827 equiv "( ( pen ua ; v0 = 3 - ul ) ; move 4 , ul ) " () () 16 0)
 (828 uniq "( move 2 , ua - ul ; move 3 , ua ; move ua , 4 ) " ((822 sub 3))
  () -1 0)
 (829 uniq "( move ua , 4 ; move ua , 3 ; move 2 , ua - ul ) "
  ((729 sub 3) (918 sub 3)) () -1 0)
 (830 uniq "( move 0 , 4 ; move ua - ul , ul ; move ua , 4 ) " () () -1 0)
 (831 uniq "( move 2 , 4 ; move ul , 2 ; move ua + ul , ul ) " ((840 sub 3))
  () -1 0)
 (832 uniq "( move ua , ul / 3 ; move ul , 0 ; move 3 , ul ) " () () -1 0)
 (833 uniq "( move ua , ua - 4 ; move ua , 2 ; v0 = ua ) " () () -1 0)
 (834 uniq "( move ua - ul , 2 ; move 4 , 3 ; move ua , ul ) " ((931 sub 3))
  () -1 0)
 (835 uniq "( move ua , 2 ; move 4 , 4 ; move ua , ua - ul ) " () () -1 0)
 (836 equiv "( move 3 + ul , 2 ; v0 = ua ; move ua , ul ) " () () 40 0)
 (837 equiv "( move ua , ul - 2 ; move 4 , ua ; v0 = ua ) " () () 2 0)
 (838 equiv "( pen 4 + 3 ; pen ua ; move 3 + ul , ua * 4 ) " () () 12 0)
 (839 uniq "( move 4 , ul ; move ul , 3 ; move ua , 2 - ul ) " () () -1 0)
 (840 uniq "( move 3 , 4 ; move 3 , ul ; move ua + ul , ul ) "
  ((831 sub 3) (919 sub 3)) () -1 0)
 (841 uniq "( move ua , ul ; move 3 , 3 - ul ; move 4 , ul ) " () () -1 0)
 (842 equiv "( move 4 , ul ; move ua , ua ; v0 = 2 - ul ) " ((213 del 5)) ()
  1 0)
 (843 equiv "( move ua , ua ; ( move ua , 4 ; v0 = ua ) ) " ((780 sub 1)) ()
  33 0)
 (844 uniq "( move ua + 4 , ul ; pen ua + ua ; move 2 , 2 ) " () () -1 0)
 (845 uniq "( pen 2 ; move ua + 2 , ua + ua ; move ul , 4 ) " () () -1 0)
 (846 equiv "( pen 4 ; move ua + ul , 2 ; move 2 , ua + ua ) " () () 461 0)
 (847 uniq "( move ua + ul , 0 ; pen 4 ; move 2 , ul + ul ) " () () -1 0)
 (848 uniq "( move 0 - ua , ul ; pen 3 + ul ; move 2 , ul ) " () () -1 0)
 (849 uniq "( move ua , 4 ; move ua + 3 , ua + 2 ; pen ua ) " () () -1 0)
 (850 uniq "( pen ul ; move ua , 2 - ua ; move 3 , ua - 2 ) " () () -1 0)
 (851 uniq "( pen 2 ; move 2 - ua , ul ; move 2 - ua , ul ) " () () -1 0)
 (852 equiv "( ( move 3 , 0 ; v0 = 2 ) ; move ua , v0 ) " () () 34 0)
 (853 uniq "( pen ua ; move ul , ua + 2 ; move 3 + ul , 4 ) " () () -1 0)
 (854 uniq "( pen ul ; move ua , 3 ; move ua + 2 , ua - 4 ) " () () -1 0)
 (855 equiv "( pen ua + ul ; ( move ua , 0 ; move 2 , ua ) ) " () () 69 0)
 (856 uniq "( pen ua ; move 2 , ul + ul ; move 4 , ua - 2 ) " () () -1 0)
 (857 uniq "( pen ua ; move 4 , 3 - ua ; move ua + ul , 2 ) " () () -1 0)
 (858 uniq "( pen ua + ul ; move 2 + 2 , 4 ; move ul , ul ) " () () -1 0)
 (859 equiv "( pen 3 ; move ul / ua , 0 ; move ua , 3 + ul ) " () () 13 0)
 (860 uniq "( pen 4 ; move ua + 2 , ua ; move ua , 4 - ua ) " ((199 del 5))
  () -1 0)
 (861 uniq "( move 0 - ua , 4 ; pen ua ; move ul , 4 - ua ) " () () -1 0)
 (862 uniq "( move 0 - ua , 3 / 2 ; pen ua ; move ul , ul ) " () () -1 0)
 (863 uniq "( pen 4 + ul ; move ua - 2 , ua ; move 3 , ul ) " ((259 del 4))
  () -1 0)
 (864 uniq "( move ua , ua * 3 ; pen 3 ; move 2 - ua , ul ) " () () -1 0)
 (865 uniq "( move ua + 2 , ul - 4 ; pen 2 ; move ul , ul ) " () () -1 0)
 (866 equiv "( pen ua + ul ; move ul , 3 - 2 ; move 4 , ul ) " () () 328 0)
 (867 uniq "( ( pen ua ; move ul - ua , 3 ) ; move ua , 0 ) " () () -1 0)
 (868 equiv "( pen ua ; ( move ua , ua ; move 2 + 2 , ul ) ) " () () 645 0)
 (869 uniq "( move ua , ua - 4 ; pen ua ; move 3 , 4 - ua ) " () () -1 0)
 (870 uniq "( move ua , ua - 3 ; pen 3 ; move ul + ul , 4 ) " ((961 sub 3))
  () -1 0)
 (871 uniq "( pen 4 ; move ua + ul , 3 + ul ; move ua , 4 ) " () () -1 0)
 (872 equiv "( pen ua * 2 ; v0 = ua ; move ua , ua * 2 ) " () () 14 0)
 (873 uniq "( pen ua / 2 ; move 2 , ul ; move ua , 3 - ul ) " () () -1 0)
 (874 uniq "( pen ua - 3 ; move 3 , ua ; move ul , 3 - ul ) " ()
  ((1139 nul 7)) -1 0)
 (875 uniq "( move ul , 2 ; pen 4 ; move ua - 3 , ua - ul ) " () () -1 0)
 (876 uniq "( pen 3 * 3 ; move 2 , ul ; move ua + ul , ul ) " () () -1 0)
 (877 uniq "( pen 4 + 2 ; move 3 , ul ; move ul - ua , ul ) " () () -1 0)
 (878 uniq "( pen ua - ul ; move ua - ul , 3 ; move 4 , 3 ) " () () -1 0)
 (879 uniq "( move ua , ua ; ( pen ua ; move 2 , 3 + ul ) ) " ((884 sub 2))
  () -1 0)
 (880 uniq "( move ua , ul ; pen 3 ; move ul - 3 , 4 + ul ) " () () -1 0)
 (881 uniq "( move ua , ul ; move ua , 3 * 3 ; pen ul - 3 ) " () () -1 0)
 (882 uniq "( pen ua * 3 ; move ua + ul , 4 ; move 4 , ua ) " () () -1 0)
 (883 uniq "( pen 3 + ul ; move ua / 4 , 2 ; move ua , ul ) " () () -1 0)
 (884 equiv "( move ua , 3 ; ( pen ua ; move 2 , ua + ul ) ) "
  ((287 del 4) (879 sub 2)) () 287 0)
 (885 uniq "( move ua , 3 / ua ; pen 4 ; move 2 , ua - ul ) " () () -1 0)
 (886 uniq "( move ua - 2 , ul ; pen 4 ; move 4 , ua * ua ) " () () -1 0)
 (887 uniq "( pen 4 ; move ua , 3 ; move ul - 4 , ua + ul ) " ((299 del 4))
  () -1 0)
 (888 equiv "( pen 2 * 2 ; move ua , ul - ua ; v0 = ul ) " () () 17 0)
 (889 equiv "( ( move ua , 0 ; pen ua + 4 ) ; move ul , ul ) " ((975 sub 3))
  () 651 0)
 (890 equiv "( pen ul / ua ; move ul - ua , 3 ; move 3 , 3 ) " () () 0 0)
 (891 uniq "( move ua + ul , ul ; pen 4 ; move ua / 2 , 4 ) " () () -1 0)
 (892 equiv "( move 4 , ul ; ( pen ua + ul ; move ul , 0 ) ) "
  ((547 del 2) (591 del 2)) () 325 0)
 (893 uniq "( pen ua ; move 4 * 2 , ul ; move 4 - ua , ul ) " () () -1 0)
 (894 equiv "( ( pen ul / ua ; move 2 , ua ) ; move ua , 2 ) " () () 0 0)
 (895 uniq "( ( pen ua * 2 ; move ua , ul ) ; move ul , 2 ) " () () -1 0)
 (896 uniq "( pen 3 + ul ; move 3 , ul / ua ; move 4 , ul ) " ((964 sub 3))
  () -1 0)
 (897 uniq "( pen 4 ; move 4 , ua * ua ; move ul , 3 - ul ) " () () -1 0)
 (898 uniq "( ( move 2 , ul ; pen 3 ) ; move ua + ul , ul ) " () () -1 0)
 (899 uniq "( move ua + ul , ul ; move 0 , ua ; move ua , 2 ) "
  ((191 del 5) (912 sub 3)) ((912 sub 3) (1018 nul 7)) -1 0)
 (900 uniq "( move ul - ua , ua ; move ul , 0 ; move ua , 2 ) " () () -1 0)
 (901 equiv "( move ua , ul ; move ua , 2 - ua ; move ul , 0 ) " () () 193 0)
 (902 uniq "( move ua , 3 ; ( pen ua - ul ; move ul , 4 ) ) " () () -1 0)
 (903 uniq "( move 4 + ul , ul ; ( pen ua ; move 3 , ul ) ) " () () -1 0)
 (904 uniq "( move ua / 2 , ul ; ( pen 3 ; move ua , ul ) ) " ((905 sub 3))
  () -1 0)
 (905 uniq "( move ul - 3 , ul ; ( pen 3 ; move ua , ul ) ) " ((904 sub 3))
  () -1 0)
 (906 uniq "( pen 3 ; ( move 4 , ul ; move ua , ua - ul ) ) " ((213 del 5))
  () -1 0)
 (907 uniq "( ( move ul , ua / 4 ; pen ua ) ; move 3 , ua ) " () () -1 0)
 (908 uniq "( ( move 4 , ua * ua ; pen ul ) ; move ua , 4 ) " () () -1 0)
 (909 equiv "( move 3 , 0 ; move ua , ua - ul ; move ul , ua ) " () () 443 0)
 (910 equiv "( move ua , ua ; move 4 , ua * ua ; move 0 , ul ) "
  ((989 sub 2)) () 159 0)
 (911 uniq "( pen 4 ; move ul / 4 , ul ; move ua , ua / 4 ) " () () -1 0)
 (912 equiv "( move 2 - ul , ul ; move ua , ua ; move ua , 2 ) "
  ((899 sub 3)) () 899 0)
 (913 equiv "( pen 4 ; ( move ua , ul - ua ; v0 = ul ) ) " () () 17 0)
 (914 equiv "( pen ua * 4 ; v0 = 4 ; move 4 + v0 , 0 ) " () () 69 0)
 (915 uniq "( move ua , ua ; move ul - ua , ua ; move 4 , 2 ) " ((167 del 5))
  () -1 0)
 (916 uniq "( move ul , ua ; move 3 - ua , 2 ; move ua , ul ) " ((988 sub 3))
  () -1 0)
 (917 equiv "( pen 2 - 3 ; pen ua * ua ; move ua - ul , 2 ) " () () 72 0)
 (918 uniq "( move ua , 4 ; move ua , 2 ; move ul , ua + ul ) "
  ((733 sub 3) (829 sub 3)) () -1 0)
 (919 uniq "( move ua , ua ; move 3 , ul ; move 3 + ul , ul ) "
  ((201 del 5) (207 del 5) (840 sub 3)) () -1 0)
 (920 uniq "( move 3 , ul ; move ua + ul , ua ; move ua , 4 ) " ((924 sub 2))
  () -1 0)
 (921 uniq "( move ua , 4 ; move ua , ul + ul ; move 2 , ul ) " () () -1 0)
 (922 equiv "( v0 = ua ; move v0 , v0 ; move v0 , 2 ) " ((929 sub 1)) () 32
  0)
 (923 equiv "( pen ul - ua ; pen 4 ; move 4 + ul , ua + 3 ) " () () 145 0)
 (924 uniq "( move ua , ul ; move ua + ul , 4 ; move ua , 4 ) "
  ((198 del 5) (920 sub 2)) () -1 0)
 (925 equiv "( pen 4 + ul ; move 3 + ul , ua ; pen 4 + ul ) " () () 12 0)
 (926 uniq "( move 3 , ul ; move ua / 2 , ul ; move ua , ul ) " () () -1 0)
 (927 uniq "( move ul , 3 ; move 4 , ul ; move ua , ua * ua ) " () () -1 0)
 (928 uniq "( move 4 , ul - 2 ; move ua , ul ; move ul , ul ) " () () -1 0)
 (929 equiv "( v0 = ua ; move v0 , v0 ; move v0 , 4 ) " ((922 sub 1)) () 33
  0)
 (930 equiv "( v0 = 4 ; move v0 , v0 ; move ua , v0 ) " () () 422 0)
 (931 uniq "( move ua , ul / ua ; move 4 , 3 ; move ua , ul ) "
  ((243 del 5) (834 sub 3) (985 sub 3)) () -1 0)
 (932 uniq "( ( pen ua ; pen ua - 4 ) ; move ua , ul - 3 ) " () () -1 0)
 (933 uniq "( move 2 + ul , 2 ; pen ua ; move ua , ua + ua ) " () () -1 0)
 (934 equiv "( ( pen ul / ua ; move 3 * 2 , ua ) ; pen ua ) " () () 0 0)
 (935 equiv "( move 2 - ua , ul ; move ua , ul - ua ; pen 0 ) " () () 605 0)
 (936 uniq "( move ua , 0 ; pen ua - 3 ; move ua , ua * ua ) " ((235 del 5))
  () -1 0)
 (937 uniq "( pen ua ; move ua , 3 + ul ; move ua , ua + 3 ) " () () -1 0)
 (938 equiv "( pen ua - ul ; move 0 , ua - 3 ; move ua , ul ) " () () 76 0)
 (939 equiv "( move 0 , ul - 4 ; pen ua ; move ua , ua + ul ) " () () 76 0)
 (940 uniq "( move ul , 0 ; pen ua ; move ua + ul , ul / 2 ) " ((955 sub 2))
  () -1 0)
 (941 uniq "( move ua , 3 ; pen 2 + ul ; move ua , ul + ul ) " () () -1 0)
 (942 uniq "( move ua , ul ; move ua + ul , ul - 3 ; pen 2 ) " ((193 del 5))
  () -1 0)
 (943 uniq "( pen ua + ua ; move ua , 4 ; move 4 , ua + ul ) " ((187 del 5))
  () -1 0)
 (944 equiv "( move 3 + ul , 4 ; v0 = ua ; move v0 , 4 ) " () () 422 0)
 (945 uniq "( move ul - ua , ua ; pen ua + ua ; move 4 , 4 ) " () () -1 0)
 (946 equiv "( move ua , ul ; pen ul ; move 3 + ul , 2 - ul ) " ((207 del 5))
  () 67 0)
 (947 equiv "( move ua , ul / ua ; move ua , 2 - ul ; pen 2 ) "
  ((203 del 5) (243 del 5)) () 243 0)
 (948 uniq "( pen ua ; move ua , 3 - 4 ; move ul , ua + ul ) " () () -1 0)
 (949 uniq "( move ua + 4 , ul ; move ul - ua , ul ; pen 2 ) " () () -1 0)
 (950 uniq "( pen ua * 3 ; move ua , 2 ; move ul - ua , ul ) " ((291 del 4))
  () -1 0)
 (951 uniq "( move ul - 2 , ul ; pen ua + 4 ; move ua , ul ) " () () -1 0)
 (952 uniq "( move 4 , ul + ul ; pen ua ; move 2 , ul + ul ) " () () -1 0)
 (953 uniq "( move ua , ul ; pen ua ; move ul - 4 , ua + 4 ) " ((351 del 4))
  () -1 0)
 (954 uniq "( pen ua ; move ua , 4 ; move ul - ua , 4 + ul ) " () () -1 0)
 (955 uniq "( move ul , ul ; pen 2 ; move ua + ul , ul / 2 ) " ((940 sub 2))
  () -1 0)
 (956 uniq "( pen ua / 2 ; move ul , ua ; move ua + 4 , ul ) " ((350 del 4))
  () -1 0)
 (957 equiv "( ( move ua , ua ; move ul - ua , ul ) ; pen 0 ) " ((241 del 5))
  () 241 0)
 (958 equiv "( move 4 , ul ; pen ua * ua ; move ua + ul , 3 ) " () () 321 0)
 (959 uniq "( move ul - ua , 2 ; pen ua * 4 ; move ul , ua ) " () () -1 0)
 (960 uniq "( pen ua + 4 ; move ua / 2 , ua ; move ul , ul ) " () () -1 0)
 (961 uniq "( move ua , ua / 3 ; pen 2 ; move ul + ul , ul ) " ((870 sub 3))
  () -1 0)
 (962 equiv "( move 4 , ua + ul ; pen ua ; move 3 - ul , ul ) " () () 540 0)
 (963 uniq "( pen ua ; move ua + ul , ul ; move 3 , ua / 4 ) " () () -1 0)
 (964 uniq "( pen ul + ul ; move 3 , ul / 2 ; move ua , ul ) " ((896 sub 3))
  () -1 0)
 (965 equiv "( pen ua ; move ua + 4 , ul ; v0 = ul + ul ) " () () 77 0)
 (966 uniq "( pen ua ; move 4 , ul + ul ; move 4 , ul + ul ) " () () -1 0)
 (967 uniq "( move ua , 4 + ul ; pen 4 ; move ul , ul + ul ) " ((329 del 4))
  () -1 0)
 (968 uniq "( move ua , 3 / 2 ; pen ua - ul ; move ul , ul ) " () () -1 0)
 (969 uniq "( move 2 - ul , ua / 4 ; pen ua ; move ua , ul ) " () () -1 0)
 (970 uniq "( move ua / 4 , 3 - ul ; move ua + 3 , 3 + 2 ) " () () -1 0)
 (971 equiv "( ( pen ua ; move ul , 3 + ul ) ; move ua , ua ) " () () 321 0)
 (972 equiv "( ( pen ua ; move 3 , ua ) ; move ua - ul , ua ) " ((316 del 4))
  () 69 0)
 (973 uniq "( move 4 , ul ; pen ua + ua ; move 4 , ul / ua ) " () () -1 0)
 (974 uniq "( pen ua ; move ul / ua , 3 / ua ; move ua , 4 ) " () () -1 0)
 (975 uniq "( ( move ua , ua ; pen ua + ul ) ; move 4 , ul ) " ((889 sub 3))
  ((1308 nul 9)) -1 0)
 (976 uniq "( ( move 2 - ua , ul ; pen ua ) ; move ul , ul ) " () () -1 0)
 (977 uniq "( ( pen ua ; move ul , ul ) ; move ua + ul , 4 ) " () () -1 0)
 (978 equiv "( move ua , ua + ul ; ( pen 4 ; move ul , ul ) ) " ((329 del 4))
  () 329 0)
 (979 equiv "( v0 = ua ; ( pen v0 ; move v0 , v0 ) ) " () () 14 0)
 (980 uniq "( move ua - ul , 4 ; ( pen ua ; move ul , ul ) ) " () () -1 0)
 (981 uniq "( move ua , ul ; move ul , ua * ua ; move ua , 0 ) "
  ((989 sub 3)) () -1 0)
 (982 equiv "( pen ua ; ( move ua , ua ; v0 = ul / ua ) ) " () () 14 0)
 (983 equiv "( ( pen ul / ua ; move ua , ul ) ; v0 = ul ) " ((664 del 2)) ()
  0 0)
 (984 uniq "( move ua , ul ; move ul , ua ; move ua + ul , 4 ) " () () -1 0)
 (985 uniq "( move ua , ul / ua ; move ul , ul ; move ua , 2 ) "
  ((203 del 5) (931 sub 3)) () -1 0)
 (986 equiv "( pen ul + ul ; move ua , 4 + 4 ; pen ul - ua ) " () () 26 0)
 (987 equiv "( move ua , ua ; move ul - ua , ul ; v0 = ul ) " ((241 del 5))
  () 241 0)
 (988 uniq "( move ul , ul ; move ul - ua , 4 ; move ua , ul ) "
  ((916 sub 3)) () -1 0)
 (989 uniq "( move ua , ul ; move 4 , ua * ua ; move ul , ul ) "
  ((910 sub 2) (981 sub 3)) () -1 0)
 (990 equiv "( pen ul / ua ; ( v0 = 0 ; move 4 , v0 ) ) " () () 0 0)
 (991 equiv "( v0 = ua + ul ; move ua , 0 ; move 4 , v0 ) " () () 35 0)
 (992 uniq "( pen ua ; move 0 - ua , ua ; move ul , ua + ul ) " () () -1 0)
 (993 uniq "( move ua , 3 ; v0 = 2 ; move ua , ul - v0 ) " () () -1 0)
 (994 uniq "( move ua , ua ; pen ua - ul ; move 2 - ua , ul ) " () () -1 0)
 (995 uniq "( move 3 , ua ; pen ua + ua ; move ul , ua - ul ) " () () -1 0)
 (996 uniq "( pen ul + ul ; move ua , ua ; move ua , 4 + ul ) " () () -1 0)
 (997 equiv "( pen ua + ua ; move ua , ua ; v0 = ua * ua ) " () () 14 0)
 (998 uniq "( pen ua ; move ua , ua - 3 ; move ul , ua * ua ) " () () -1 0)
 (999 uniq "( move ul , 4 ; pen ua + ua ; move ua + ul , ul ) " () () -1 0)
 (1000 uniq "( ( pen ua ; move 2 , 4 ; move ua , 4 ) ; pen 2 ) " () () -1 0)
 (1001 uniq "( move 3 , ua ; v0 = ul / 3 ; move ua , v0 ) " () () -1 0)
 (1002 uniq "( pen ua ; move ua , ul - ua ; move ua - ul , 4 ) "
  ((204 del 5) (320 del 4)) () -1 0)
 (1003 equiv "( move ua , ul - ua ; pen 3 + ul ; move ul , ul ) " () () 329
  0)
 (1004 uniq "( move 3 , ul + ul ; pen ua + ul ; move ul , ul ) " () () -1 0)
 (1005 equiv "( move 4 , ul + ul ; move ua , ua + ul ; pen ul ) " () () 40 0)
 (1006 uniq "( move ul , ua ; pen ua + ul ; move ua , ua / 4 ) " () () -1 0)
 (1007 uniq "( move ul - ua , ul ; pen 3 ; move ua - ul , ul ) " () () -1 0)
 (1008 uniq "( pen ua + ul ; move ul , 4 / ua ; move ua , ul ) "
  ((350 del 4)) () -1 0)
 (1009 uniq "( pen ua ; move ul , 4 / ua ; move ua , ul + ul ) "
  ((350 del 4)) () -1 0)
 (1010 uniq "( move 4 , 0 ; v0 = 4 * 2 ; move v0 , v0 ) " () () -1 0)
 (1011 equiv "( ( move ua , ua ; pen ul ) ; move ua , ul + ul ) "
  ((236 del 5)) () 32 0)
 (1012 equiv "( pen ua ; v0 = 3 ; move ua + 3 , ul - v0 ) " () () 137 0)
 (1013 uniq "( move 4 , 2 ; move ul , 4 + 3 ; move 4 + 2 , 4 ) " () () -1 0)
 (1014 uniq "( move 2 - 4 , 2 ; move ua + 2 , 4 ; v0 = 4 ) " () () -1 0)
 (1015 uniq "( move ul , ul ; move ua , ul ; move ua , ua + ul ) " ()
  ((1016 nul 4)) -1 0)
 (1016 equiv "( move ul , ul ; move ua , ua + ul ; move ua , ul ) " () ()
  1015 0)
 (1017 uniq "( pen ua - ul ; move ua + ul , ua / 4 ; pen ul ) " () () -1 0)
 (1018 equiv "( v0 = ua + ul ; move v0 , ul ; move ua , 2 ) " () () 899 0)
 (1019 equiv "( ( v0 = 4 * 2 ; pen v0 ) ; move v0 , 0 ) " () () 69 0)
 (1020 equiv "( pen ua ; ( move 4 , ul ; move 0 , ua ) ; pen 2 ) " () () 16
  0)
 (1021 equiv "( move ua , ua ; v0 = ua ; move v0 - 3 , ul ) " () () 152 0)
 (1022 uniq "( v0 = ua - ul ; move ua , 2 ; move v0 , ul ) " ()
  ((1027 nul 12)) -1 0)
 (1023 uniq "( move ua , 4 ; v0 = ul - ua ; move v0 , ul ) " () () -1 0)
 (1024 uniq "( pen ua + ul ; move ul , ul ; move ua , ua - ul ) "
  ((358 del 4)) () -1 0)
 (1025 uniq "( move ul - ua , 4 + 3 ; move ul - ua , ua + 2 ) " () () -1 0)
 (1026 uniq "( move 2 , 0 ; move 4 , 4 ; move 0 - ua , 0 - ul ) " () () -1 0)
 (1027 equiv "( move ua , ul + ul ; move ua - ul , ul ; pen ul ) "
  ((240 del 5)) () 1022 0)
 (1028 equiv "( move 4 , ul / ua ; v0 = ua ; move v0 , ul ) " () () 452 0)
 (1029 uniq "( v0 = 3 + 2 ; move ua , 4 ; move v0 , v0 ) " ((1100 sub 2)) ()
  -1 0)
 (1030 equiv "( pen ua + ul ; v0 = ua ; move v0 + 2 , ua ) " () () 69 0)
 (1031 uniq "( move 3 , 2 + ul ; move 4 , 2 ; move ua , 3 + 2 ) " () () -1 0)
 (1032 uniq "( move 3 , 0 ; move ul - ua , 4 ; move 2 , 3 * 2 ) " () () -1 0)
 (1033 equiv "( move 4 , 2 + 2 ; move ua , ua + 4 ; v0 = 2 ) " () () 422 0)
 (1034 equiv "( ( move ua , 2 ; v0 = 4 ) ; move v0 , v0 ) " () () 21 0)
 (1035 equiv "( ( pen 2 ; pen 3 ) ; move ua , 4 ; pen ua + ua ) " () () 13 0)
 (1036 equiv "( v0 = 2 + 2 ; pen v0 ; move ua , v0 + 2 ) " () () 25 0)
 (1037 equiv "( ( v0 = 4 ; move v0 , 4 ) ; move ua , v0 ) " () () 422 0)
 (1038 equiv "( pen 3 + ul ; v0 = ul ; move ua / v0 , ul ) " () () 17 0)
 (1039 uniq "( move ul - ua , 4 / 3 ; move 3 , 2 ; move 3 , 3 ) " () () -1 0)
 (1040 uniq "( move ua , ua ; move 2 / 4 , 4 ; move 3 , 4 - 3 ) " () () -1 0)
 (1041 equiv "( v0 = 4 * 2 ; pen v0 ; move ul + v0 , 0 ) " () () 69 0)
 (1042 uniq "( move 4 , 4 * 2 ; move 4 , ul ; move 3 , ua + 4 ) " () () -1 0)
 (1043 uniq "( move 0 , ua ; pen 4 - 2 ; move ua + 2 , 3 * 2 ) " () () -1 0)
 (1044 uniq "( ( move ul , 2 / 4 ; move 3 , 3 ) ; move ua , 2 ) " () () -1 0)
 (1045 uniq "( pen ua ; v0 = 4 ; move 4 + 2 , v0 * v0 ) " () () -1 0)
 (1046 equiv "( ( pen ua ; move ua , 2 ; move ul , 0 ) ; pen ul ) " () () 70
  0)
 (1047 uniq "( move 0 - 4 , 2 + 2 ; pen ul ; move ua , 4 / 3 ) " () () -1 0)
 (1048 uniq "( move 3 + 3 , 2 ; ( move ua , 3 + ul ; pen 2 ) ) " () () -1 0)
 (1049 uniq "( ( move 3 * 2 , 3 ; move ua , 3 + 2 ) ; pen ua ) " ()
  ((1346 nul 15)) -1 0)
 (1050 uniq "( move ua / 2 , ua + ul ; move ua + ul , ul - 2 ) "
  ((193 del 6) (435 del 4)) () -1 0)
 (1051 uniq "( move ua , ua ; pen 3 ; ( pen 4 ; move ul , ua ) ) " () () -1
  0)
 (1052 uniq "( ( move 4 + 3 , ul ; pen 3 + 3 ) ; move ua , 2 ) " () () -1 0)
 (1053 uniq "( pen ua ; ( move ua , 4 ; move ua , 4 ; pen ul ) ) " () () -1
  0)
 (1054 uniq "( move ua , ua + 2 ; move ua , 4 + 3 ; move 0 , 3 ) " () () -1
  0)
 (1055 uniq "( pen 4 ; ( move ua + 2 , ua ; move 4 + 3 , 4 ) ) " () () -1 0)
 (1056 uniq "( pen 3 ; ( move 3 , ua - 2 ; move ua , 4 - 3 ) ) " () () -1 0)
 (1057 uniq "( move ua , 2 + 2 ; move 4 + ul , 2 ; move 2 , ul ) " () () -1
  0)
 (1058 equiv "( move 0 , 3 ; ( move ua , 0 ; move ua , 2 + ul ) ) " () () 36
  0)
 (1059 uniq "( move ua , 2 ; move 3 + ul , 4 - ua ; move 3 , 2 ) " () () -1
  0)
 (1060 uniq "( move 3 , ul ; move 4 + ul , 2 ; move 3 , 2 + ul ) " () () -1
  0)
 (1061 uniq "( move 4 + ul , ua ; move ua , 3 / 2 ; v0 = 0 ) " ()
  ((1422 nul 12)) -1 0)
 (1062 uniq "( move 4 , ul ; move 3 - ul , 4 ; move ua + 2 , 2 ) " () () -1
  0)
 (1063 equiv "( move ua , ua + 2 ; ( move 4 , 4 ; move 0 , ua ) ) " () () 21
  0)
 (1064 uniq "( move ua , 3 * 2 ; v0 = 4 ; move ua , 0 - ul ) " () () -1 0)
 (1065 uniq "( move ua , 3 ; move ua - 2 , 2 ; v0 = ua / 3 ) " () () -1 0)
 (1066 equiv "( move 4 , ul ; ( v0 = ua ; move v0 , v0 ) ) " () () 1 0)
 (1067 uniq "( ( move ua , ua + 2 ; move 2 , 3 ) ; move ua , 4 ) " () () -1
  0)
 (1068 uniq "( move 3 , 4 + 4 ; move ua , ua + 3 ; move 4 , ul ) " () () -1
  0)
 (1069 uniq "( move ua , ul ; move 3 / 2 , ul ; move 3 * 3 , 2 ) " () () -1
  0)
 (1070 equiv "( ( move ul - ua , 0 ; move ua , 2 ) ; v0 = 3 ) " () () 433 0)
 (1071 uniq "( move 3 + ul , 3 ; move ua , 2 ; v0 = 4 - ul ) " () () -1 0)
 (1072 uniq "( move 2 - ua , 3 ; move ua , 3 / 2 ; v0 = ul ) " () () -1 0)
 (1073 uniq "( ( move ua , 2 + ul ; move 2 , 4 ) ; move 4 , ul ) "
  ((1132 sub 2)) () -1 0)
 (1074 uniq "( move 4 , 2 ; ( move ua + ul , ul ; move 4 , 2 ) ) " () () -1
  0)
 (1075 uniq "( move 4 , 4 ; move ul , 4 ; move ua + 4 , 4 - ua ) " () () -1
  0)
 (1076 uniq "( ( move 2 , ua / 2 ; move 4 , 2 ) ; move ua , ul ) " () () -1
  0)
 (1077 uniq "( pen 2 ; move 2 , 0 - ua ; move 4 + 3 , ul + ul ) " () () -1 0)
 (1078 uniq "( pen 4 + 2 ; move ua , 3 + 3 ; move ua + ul , 2 ) " () () -1 0)
 (1079 equiv "( pen ua + 3 ; ( move 3 , 0 ; move ua - ul , 0 ) ) " () () 69
  0)
 (1080 uniq "( pen 4 ; move 0 - 4 , ul - 2 ; move 2 , ua + ul ) " () () -1 0)
 (1081 uniq "( move 2 , 4 + 2 ; pen ua * 3 ; move 2 , ua - ul ) " () () -1 0)
 (1082 uniq "( v0 = ua + ul ; pen 4 ; move v0 , v0 / 4 ) " () () -1 0)
 (1083 uniq "( pen ua ; move 3 + ul , 3 + 2 ; move 4 - 3 , ul ) " () () -1 0)
 (1084 uniq "( move 0 - 4 , ua + ul ; pen 4 * 3 ; move ua , 3 ) " () () -1 0)
 (1085 uniq "( move 4 / 2 , ua ; pen 3 + 3 ; move 3 + ul , ua ) " () () -1 0)
 (1086 uniq "( move ua , ua ; move 0 , ua ; move ua + 3 , 2 + 2 ) " () () -1
  0)
 (1087 uniq "( pen ua + 3 ; ( move 3 , 3 ; move 2 , ua + ul ) ) " () () -1 0)
 (1088 uniq "( pen 3 * 2 ; move 2 , ul / 3 ; move 3 + ul , ul ) " () () -1 0)
 (1089 uniq "( ( move 0 - 4 , 3 - ul ; pen 2 ) ; move ua , ul ) "
  ((1158 sub 3)) () -1 0)
 (1090 uniq "( move ua , ua - ul ; pen 4 * 3 ; move 3 , 3 - 4 ) " () () -1 0)
 (1091 uniq "( move ua , ul ; v0 = ua + 2 ; move v0 , v0 ) " ()
  ((1373 nul 12)) -1 0)
 (1092 uniq "( pen ua ; ( move 2 + ul , 4 + 3 ; move ua , 4 ) ) " () () -1 0)
 (1093 uniq "( move 4 - ul , ua / 4 ; move ua + 2 , 3 ; pen 4 ) " () () -1 0)
 (1094 uniq "( move ua , 3 ; ( pen 4 + 2 ; move ul + ul , 3 ) ) " () () -1 0)
 (1095 uniq "( v0 = ua + 2 ; move v0 , ul ; move ua , v0 ) " () () -1 0)
 (1096 uniq "( ( pen ua ; move 4 + 2 , ua ) ; move ul , 4 * 2 ) " () () -1 0)
 (1097 equiv "( v0 = 3 - ua ; move ua , v0 ; move ua , v0 ) " () () 36 0)
 (1098 equiv "( move ua + 2 , 3 ; move ua , 3 ; move 0 , ua - ul ) " () ()
  745 0)
 (1099 equiv "( move ul , ul ; move ul + ul , 0 ; move 4 + 3 , 0 ) " () () 1
  0)
 (1100 equiv "( v0 = 3 + ul ; move ua , ul ; move v0 , v0 ) "
  ((1029 sub 2) (1167 sub 2)) () 43 0)
 (1101 uniq "( ( pen 4 + 4 ; move ul , 3 ) ; move ua - ul , 2 ) " () () -1 0)
 (1102 uniq "( move ua , 0 ; move 0 - ua , 4 * 3 ; move ul , ua ) " () () -1
  0)
 (1103 uniq "( move 3 , 2 ; move 0 , ul ; move ua + ul , 3 + ul ) " () () -1
  0)
 (1104 equiv "( move ua , ua + ul ; move ua , 4 ; move 0 , 3 * 2 ) " () () 64
  0)
 (1105 uniq "( move ua + ul , ua ; move 3 + ul , 2 ; move 3 , 2 ) " () () -1
  0)
 (1106 uniq "( move ul , ua ; move ua , 3 + 2 ; move ua , 2 * 2 ) " () () -1
  0)
 (1107 equiv "( move 3 + ul , 4 - ua ; move ul , 0 ; move ua , 2 ) " () ()
  125 0)
 (1108 uniq "( move 3 , ua ; move 3 + ul , 4 ; move ua + ul , 0 ) " () () -1
  0)
 (1109 equiv "( ( move 0 , ua - ul ; pen 4 * 4 ) ; move 4 , ul ) " () () 12
  0)
 (1110 uniq "( move 4 , 2 ; move ul , ua + 2 ; move ua + 2 , ul ) " () () -1
  0)
 (1111 uniq "( move ua , 2 + ul ; move ua , 2 ; move ua , 4 / 2 ) " () () -1
  0)
 (1112 uniq "( move ul - ua , 3 ; move 0 - 4 , ul ; move ul , 2 ) " () () -1
  0)
 (1113 uniq "( move ua , 4 ; move 3 , 3 + ul ; move 0 - ul , ul ) " () () -1
  0)
 (1114 uniq "( move ua , 3 ; move ul + ul , ul / 2 ; move 2 , 2 ) " () () -1
  0)
 (1115 uniq "( move ua , 0 - ua ; move ua - 4 , ul ; move 4 , 4 ) " () () -1
  0)
 (1116 uniq "( move ul , ua + 4 ; move 2 , 3 ; move ua + ul , 4 ) " () () -1
  0)
 (1117 uniq "( move 2 - ua , ul ; move 4 , 2 - ua ; move 3 , ul ) " () () -1
  0)
 (1118 uniq "( move ul , ul ; move ua , 4 * 3 ; move ua + 2 , 3 ) " () () -1
  0)
 (1119 uniq "( move 4 , ua * 3 ; move ul , ua ; move ua + 2 , 4 ) " () () -1
  0)
 (1120 uniq "( move 3 + ul , 3 + ul ; move ul , 4 ; move ua , 3 ) " () () -1
  0)
 (1121 uniq "( move 3 , ua ; move 3 , 2 - ul ; move 4 + ul , ul ) " () () -1
  0)
 (1122 uniq "( move ua , ul ; ( move ua + ul , 4 ; move 0 , 3 ) ) " () () -1
  0)
 (1123 equiv "( ( pen ul ; v0 = ua ; pen 3 ) ; move v0 , 4 ) " () () 13 0)
 (1124 uniq "( move ul , ul / 2 ; move ua + 3 , 2 ; v0 = ul ) " () () -1 0)
 (1125 uniq "( move 4 , 3 ; move ul , 3 ; move ul - ua , ua - 4 ) " () () -1
  0)
 (1126 uniq "( ( move ul , 0 ; move ua , ul ) ; move 4 + ul , 4 ) " () () -1
  0)
 (1127 uniq "( move 4 , ua / 2 ; move ul , ul ; move ua , 4 * 2 ) " () () -1
  0)
 (1128 uniq "( pen ua ; move 3 , 2 - ua ; move 0 - ua , 3 - ua ) " () () -1
  0)
 (1129 uniq "( move ua , ul ; ( move 4 , 3 ; move 4 , ua + ul ) ) " () () -1
  0)
 (1130 equiv "( ( move ua , ul - 4 ; move ua , ul ) ; v0 = 2 ) " () () 156 0)
 (1131 uniq "( move ua , ul / 4 ; v0 = 2 - ul ; move 4 , ul ) " () () -1 0)
 (1132 uniq "( ( move ua , 3 + ul ; move ul , 4 ) ; move 4 , ul ) "
  ((1073 sub 2)) () -1 0)
 (1133 uniq "( move 4 , ua ; ( move ua + 4 , 4 ; move ul , ul ) ) " () () -1
  0)
 (1134 uniq "( pen 4 + ul ; move ul , ua + ul ; move 3 + 2 , 2 ) " () () -1
  0)
 (1135 uniq "( move ul , ua ; ( move ua , ul / 3 ; move 3 , 4 ) ) " () () -1
  0)
 (1136 uniq "( v0 = ua + ul ; pen ua ; move v0 , 4 / v0 ) " () () -1 0)
 (1137 uniq "( move 2 * 2 , ua ; pen ua ; move ua / 3 , 0 - ul ) " () () -1
  0)
 (1138 uniq "( pen ua ; move 0 - ua , 4 ; move ul - 3 , 2 / ua ) " () () -1
  0)
 (1139 equiv "( pen ua / 2 ; move 3 , ua * 2 ; move ul , ua + 2 ) "
  ((316 del 6)) () 874 0)
 (1140 uniq "( move 0 - ua , 4 + ul ; pen ua ; move ul , 4 + 4 ) " () () -1
  0)
 (1141 uniq "( pen ua - 4 ; move 3 + 2 , ua ; move ua - 3 , ul ) "
  ((259 del 6)) () -1 0)
 (1142 uniq "( move ul , 0 ; pen 3 + 2 ; move ua - ul , 4 / ua ) " () () -1
  0)
 (1143 uniq "( pen 4 ; move 2 - ul , ua + 3 ; move ua , 4 + ul ) " () () -1
  0)
 (1144 uniq "( pen 2 ; move 2 + ul , ul / 3 ; move 3 , ua + ul ) " () () -1
  0)
 (1145 uniq "( move ua - 4 , ua ; pen 4 + ul ; move 4 + 2 , ul ) " () () -1
  0)
 (1146 equiv "( v0 = ul ; pen ua * v0 ; move 4 / v0 , ul ) " () () 16 0)
 (1147 equiv "( move ua / 3 , 3 + ul ; pen ua ; move 4 , ua - 3 ) "
  ((294 del 6)) () 16 0)
 (1148 uniq "( pen 4 ; move 4 , 3 - ul ; move ul - ua , 2 - ua ) " () () -1
  0)
 (1149 uniq "( pen ua ; ( move 4 + 3 , ul ; move ua , ua + 3 ) ) " () () -1
  0)
 (1150 uniq "( pen 2 ; ( move ua , ul / 3 ; move ua , ua + 2 ) ) " () () -1
  0)
 (1151 uniq "( ( pen ua ; move 4 , 2 / ua ) ; move ua , ua + 2 ) " () () -1
  0)
 (1152 uniq "( move ua - ul , 4 / 3 ; pen 2 ; move 3 + ul , ul ) " () () -1
  0)
 (1153 equiv "( move ua , 0 ; move 3 + ul , ua + ul ; move 0 , ul ) " () ()
  35 0)
 (1154 uniq "( pen 3 + 3 ; ( move ua , ua * ua ; move 2 , ul ) ) " () () -1
  0)
 (1155 uniq "( move ua , 0 ; move 3 + ul , ul ; move 0 - ua , ua ) " () () -1
  0)
 (1156 uniq "( ( move ul - 2 , ul ; pen ua + 2 ) ; move 4 , ul ) "
  ((297 del 6)) () -1 0)
 (1157 uniq "( ( move 4 - ul , 2 ; pen ua ) ; move 2 , ua - ul ) " () () -1
  0)
 (1158 uniq "( ( move ua - 2 , 3 - ul ; pen 3 ) ; move ua , ul ) "
  ((1089 sub 3)) () -1 0)
 (1159 uniq "( move 2 , 2 ; ( pen ua - ul ; move 4 - ul , ua ) ) " () () -1
  0)
 (1160 equiv "( move 2 + 2 , 0 ; move ua , ua - ul ; move ul , ua ) " () ()
  449 0)
 (1161 uniq "( pen ua ; ( move ul + ul , 4 + ul ; move 4 , 2 ) ) " () () -1
  0)
 (1162 uniq "( pen ua ; ( move ua / 4 , 2 - ul ; move ua , 2 ) ) " () () -1
  0)
 (1163 uniq "( move ua - ul , 0 ; move ul , 2 ; move ua , ua + 3 ) " () () -1
  0)
 (1164 uniq "( move ua + ul , 0 ; move 3 - ua , ua ; move 2 , ul ) " () () -1
  0)
 (1165 uniq "( pen ua * 3 ; ( move 4 , ua ; move 2 - ua , ul ) ) " () () -1
  0)
 (1166 uniq "( ( move 3 , ul / 2 ; pen ua * 2 ) ; move ua , ua ) " () () -1
  0)
 (1167 uniq "( v0 = ua - ul ; move ua , ul ; move v0 , v0 ) " ((1100 sub 2))
  () -1 0)
 (1168 uniq "( ( move 4 , ul ; pen ua ) ; move 4 - ua , ul - 2 ) " () () -1
  0)
 (1169 uniq "( move ua + ul , 0 ; move 2 , ua / 2 ; move ua , ul ) " () () -1
  0)
 (1170 equiv "( move 2 , 3 ; move ua + ul , ua + ul ; move 2 , ua ) "
  ((1270 sub 3)) () 618 0)
 (1171 uniq "( pen 3 ; ( move ul - 3 , ul ; move 4 + ul , ul ) ) " () () -1
  0)
 (1172 uniq "( move ua , ua * ua ; move 2 - ua , 4 ; move 2 , ua ) " () () -1
  0)
 (1173 equiv "( move 4 , ua - ul ; move ua , ua ; move 0 , 4 - ua ) " () () 2
  0)
 (1174 equiv "( move ua , ua * 4 ; move 4 , ul - ua ; move 0 , ua ) " () ()
  35 0)
 (1175 uniq "( move 3 , ul - ua ; move ua , 2 ; move ua + ul , 4 ) " () () -1
  0)
 (1176 uniq "( move ua , 4 ; move 4 , 0 ; move ul + ul , ul + ul ) " () () -1
  0)
 (1177 uniq "( move 4 , ul ; move 0 - 4 , ul + ul ; move ua , ua ) " () () -1
  0)
 (1178 uniq "( move ul , 2 - ua ; move ua , ul ; move ua + 4 , 4 ) " () () -1
  0)
 (1179 uniq "( move ua , 4 + ul ; move 4 , ua * ua ; move 2 , ua ) " () () -1
  0)
 (1180 uniq "( move ua , 3 ; move 0 - ul , ul / ua ; move ua , 4 ) " () () -1
  0)
 (1181 uniq "( move 3 , ua ; move ua , 4 - 3 ; move ul + ul , ua ) " () () -1
  0)
 (1182 equiv "( move 2 , ua ; move 3 , ua * 4 ; move ua , ua - ul ) " () ()
  311 0)
 (1183 equiv "( move ua , ua * 2 ; move ua , ul + ul ; move 4 , 4 ) " () ()
  741 0)
 (1184 uniq "( move ul , 4 ; move ua + ul , 3 + ul ; move ua , 4 ) " () () -1
  0)
 (1185 uniq "( move ua , 2 ; ( move ul , 0 ; move ul - ua , ul ) ) " () () -1
  0)
 (1186 equiv "( ( pen ul ; pen 4 ; v0 = ul ) ; move ua , v0 ) " () () 17 0)
 (1187 uniq "( move ul , 4 ; move 4 , ua + ul ; move ua , 4 + ul ) " () () -1
  0)
 (1188 uniq "( move 3 , ul - 3 ; move ua + ul , ul ; move ul , 4 ) " () () -1
  0)
 (1189 uniq "( move 4 , 2 ; move ua + ul , ul ; move ul - 4 , ul ) " () () -1
  0)
 (1190 equiv "( pen 3 + ul ; ( pen ua + 2 ; move ul - ua , 4 ) ) " () () 85
  0)
 (1191 uniq "( move ua + ul , 2 / ua ; move ul , 4 ; move 4 , ul ) " () () -1
  0)
 (1192 uniq "( move ua , 4 ; move ua + ul , 4 - ua ; v0 = ul ) " () () -1 0)
 (1193 equiv "( ( move ul , 2 ; move ua , ua ) ; move 4 , ua + ul ) " () ()
  403 0)
 (1194 uniq "( move ua , ul ; ( move 3 , ua + ul ; move 3 , ul ) ) " () () -1
  0)
 (1195 equiv "( move ua , ua ; ( move ua , ua + 4 ; v0 = ul ) ) "
  ((566 del 4)) () 33 0)
 (1196 equiv "( pen ua / 2 ; move 0 , ua + ul ; move ua , ua - 2 ) " () () 83
  0)
 (1197 uniq "( move 4 , ul ; move 4 / ua , ul ; move ua , 3 - ul ) " () () -1
  0)
 (1198 uniq "( ( move 2 , ul ; move 4 + ul , ul ) ; move ua , ul ) " () () -1
  0)
 (1199 equiv "( ( move 4 - ul , 2 ; move ul , ua ) ; move ua , ul ) " () ()
  40 0)
 (1200 equiv "( move ua , ul - ua ; move 4 , ua ; v0 = ul / 4 ) " () () 1 0)
 (1201 uniq "( move 3 , ua ; pen ua + ua ; move ua + 3 , ul - 2 ) " () () -1
  0)
 (1202 uniq "( move ua , 4 / ua ; v0 = ul ; move 4 + ul , ul ) " () () -1 0)
 (1203 uniq "( move ul , ua ; pen 2 + ul ; move 0 - ua , 3 / ua ) " () () -1
  0)
 (1204 uniq "( pen ua + ul ; move 3 + ul , ua ; move 2 - 3 , ul ) " () () -1
  0)
 (1205 uniq "( pen ua ; move 3 , ua + ua ; move ua - 4 , ua + 4 ) " () () -1
  0)
 (1206 uniq "( pen ua ; move 3 - ua , ua ; move 2 + ul , ul - 3 ) " () () -1
  0)
 (1207 uniq "( move ua - 4 , ul - 3 ; pen ua ; move ua , ua + 2 ) " () () -1
  0)
 (1208 uniq "( move 0 - 4 , ul ; pen 3 - ul ; move ua + ul , ul ) " () () -1
  0)
 (1209 uniq "( pen ua ; move ua + 2 , ua - 3 ; move ul , ua / 4 ) "
  ((320 del 6)) () -1 0)
 (1210 uniq "( pen ua ; move ua / 3 , 2 + ul ; move ua , ua - 4 ) " () () -1
  0)
 (1211 uniq "( pen ua + 3 ; move ul - ua , 3 ; move ua , 3 / ua ) " () () -1
  0)
 (1212 uniq "( ( pen ua + ua ; move 3 , 0 ) ; move ul - ua , ul ) " () () -1
  0)
 (1213 uniq "( move ua - ul , ul - 2 ; pen 4 + 4 ; move ul , ua ) " () () -1
  0)
 (1214 equiv "( pen ua * 3 ; ( move 4 , 0 ; pen 3 ) ; v0 = 3 ) " () () 12 0)
 (1215 uniq "( move 3 / ua , ul - ua ; pen ua ; move ua , 4 * 2 ) " () () -1
  0)
 (1216 uniq "( move 3 , ua - ul ; pen ua * ua ; move ua - 4 , 4 ) " () () -1
  0)
 (1217 uniq "( pen 2 ; move ua , 2 ; ( move 0 , 4 ; move ua , 0 ) ) " () ()
  -1 0)
 (1218 uniq "( pen 3 + ul ; ( move ua + ul , ua ; move ua , 4 ) ) " () () -1
  0)
 (1219 uniq "( pen ul + ul ; move 4 * 2 , 4 ; move ul - ua , ul ) " () () -1
  0)
 (1220 uniq "( pen 3 ; ( move ua - ul , ua - ul ; move 2 , ul ) ) " () () -1
  0)
 (1221 uniq "( ( pen 2 ; move 3 , 4 ) ; ( pen 2 ; move ua , 4 ) ) " () () -1
  0)
 (1222 equiv "( move ua + ul , ul ; ( move 4 - ul , ul ; pen 2 ) ) " () ()
  206 0)
 (1223 uniq "( ( move ua , ul ; move ua , ul - 3 ) ; pen 4 + ul ) " () () -1
  0)
 (1224 equiv "( move ua , ua + ul ; move 0 , 3 ; move ua , ul - ua ) "
  ((552 del 4)) () 64 0)
 (1225 equiv "( move 2 - ua , ul - ua ; move 0 , ul ; move ua , ua ) " () ()
  605 0)
 (1226 uniq "( move ul , ua ; move 3 , ua ; move ua + ul , ua - 2 ) " ()
  ((1343 nul 14)) -1 0)
 (1227 uniq "( move ul , 4 - ul ; ( pen ua ; move ua , ul / 2 ) ) "
  ((289 del 6)) () -1 0)
 (1228 uniq "( ( move ua , 2 ; move 3 , 2 ; pen 2 ) ; move 4 , ua ) " () ()
  -1 0)
 (1229 uniq "( ( move 3 , ua ; move 0 , 4 ) ; pen 3 ; move ua , 3 ) " () ()
  -1 0)
 (1230 uniq "( move ua , ua * ua ; move 4 , ul ; move ul + ul , 0 ) " () ()
  -1 0)
 (1231 equiv "( move ua - ul , 2 + ul ; move ul , ua ; move ua , 3 ) " () ()
  36 0)
 (1232 equiv "( move ul , ua - ul ; move 4 - ul , 0 ; move ua , ua ) " () ()
  2 0)
 (1233 equiv "( v0 = 3 ; move ua , v0 + v0 ; move ua , v0 ) " () () 524 0)
 (1234 uniq "( move ul , 2 ; move ua , ua / 2 ; move ua - ul , ul ) " () ()
  -1 0)
 (1235 uniq "( move 4 , ul ; move ua + ul , 3 ; move ul , ua + ul ) " () ()
  -1 0)
 (1236 uniq "( pen 3 ; ( move 3 , ul ; move ua , 3 ) ; move 4 , 2 ) " () ()
  -1 0)
 (1237 uniq "( move ua , ul + ul ; move ul , ul ; move 4 + 3 , ul ) " () ()
  -1 0)
 (1238 equiv "( move ua - ul , ua * 2 ; move ua , ul ; v0 = ua ) "
  ((562 del 4)) () 239 0)
 (1239 uniq "( move ua , ul ; move 4 , ul ; move ua + ul , 4 - ua ) " () ()
  -1 0)
 (1240 uniq "( move ua , ul + ul ; move 4 / ua , 2 ; move ua , ul ) " () ()
  -1 0)
 (1241 uniq "( move 4 , 3 + ul ; move ul - ua , ul ; move ua , ul ) " () ()
  -1 0)
 (1242 uniq "( move 3 , 3 ; pen 4 ; ( move 3 , 4 ; move ua , ua ) ) " () ()
  -1 0)
 (1243 uniq "( move ua , ua ; move ua , 4 - ua ; move ul , ua / 4 ) " () ()
  -1 0)
 (1244 uniq "( move ua , ul / 4 ; move 3 - ua , ul ; move ua , ul ) " () ()
  -1 0)
 (1245 uniq "( move ua / 2 , 3 / ua ; move ua , ul ; move ul , ul ) " () ()
  -1 0)
 (1246 uniq "( move ul + ul , ua + 3 ; pen ua ; move ua - ul , 0 ) " () () -1
  0)
 (1247 uniq "( move ul , 4 / ua ; move ua , ul / ua ; move 3 , ul ) " () ()
  -1 0)
 (1248 uniq "( move 3 , ul / ua ; move ul , ul / ua ; move ua , 4 ) " () ()
  -1 0)
 (1249 uniq "( pen ua * 2 ; move ua , 0 - ua ; move ua , ul + ul ) " () () -1
  0)
 (1250 uniq "( move 4 , ua ; v0 = 2 ; move ua + 3 , ua - v0 ) " () () -1 0)
 (1251 uniq "( pen ua + ul ; move ua , ua - 4 ; move ul - ua , 0 ) "
  ((1263 sub 3)) () -1 0)
 (1252 uniq "( move ua , 0 - ul ; pen ua + ul ; move 3 + ul , ul ) " () () -1
  0)
 (1253 equiv "( move ua , 2 - ul ; move 3 + ul , ua + ul ; pen ua ) " () ()
  67 0)
 (1254 equiv "( pen ua ; v0 = ua ; move v0 + 2 , v0 + v0 ) " () () 69 0)
 (1255 uniq "( pen ua ; move ua - 2 , ua + 4 ; move ul + ul , ul ) " () () -1
  0)
 (1256 uniq "( move ua , 4 * 2 ; pen ua + ua ; move ua , ul + ul ) " () () -1
  0)
 (1257 equiv "( move 0 , ua ; ( v0 = ua + 4 ; move v0 , ul ) ) " () () 1 0)
 (1258 uniq "( move ua + ul , ua * ua ; move ua , ua - 3 ; pen 4 ) " () () -1
  0)
 (1259 uniq "( pen ua + 2 ; move ua / 3 , ul / ua ; move ua , ua ) " () () -1
  0)
 (1260 uniq "( pen 3 ; ( move 3 , ua ; pen ua * 2 ) ; move 4 , 4 ) " () () -1
  0)
 (1261 uniq "( ( move 2 , 4 ; pen 4 ) ; pen 4 ; move ua , ua + 4 ) " () () -1
  0)
 (1262 uniq "( v0 = 4 * 2 ; move ul - 4 , ul ; move v0 , ul ) " () () -1 0)
 (1263 uniq "( pen ua + ul ; move 4 , ua * 4 ; move ul - ua , ul ) "
  ((1251 sub 3)) () -1 0)
 (1264 uniq "( move 0 , 3 ; move 3 * 2 , 2 ; move 3 + 3 , ua / 3 ) " () () -1
  0)
 (1265 equiv "( ( pen ua ; move ua + ul , ul + ul ) ; v0 = ul ) " () () 70 0)
 (1266 uniq "( pen 3 ; ( move 3 , ua ; move ul , 2 ; move ul , 0 ) ) " () ()
  -1 0)
 (1267 uniq "( v0 = 4 + ul ; pen ua + ua ; move v0 , 3 / 2 ) " () () -1 0)
 (1268 equiv "( pen ul / ua ; move ua + ul , ua - 3 ; pen ua + 2 ) " () () 0
  0)
 (1269 uniq "( move ua + ul , ua * ua ; move 2 , ul ; move ua , ul ) " () ()
  -1 0)
 (1270 equiv "( move 4 , ua ; move ua + ul , ua + ul ; move ul , ua ) "
  ((1170 sub 3)) () 115 0)
 (1271 uniq "( move 3 , 3 ; move ua , ul ; ( pen ul ; move 2 , 2 ) ) " () ()
  -1 0)
 (1272 uniq "( move ua , ul / ua ; move ua , ul + ul ; move 3 , ua ) " () ()
  -1 0)
 (1273 uniq "( move ul - ua , ul ; move 3 , ua ; move ul - ua , ul ) " () ()
  -1 0)
 (1274 equiv "( pen 4 ; ( move ua , ul ; move 0 , 3 ; v0 = ul ) ) " () () 17
  0)
 (1275 uniq "( move 3 , ul ; move ua , 2 ; ( pen 4 ; move 3 , ul ) ) " () ()
  -1 0)
 (1276 uniq "( move 3 , ul ; pen 3 ; ( move 2 , ul ; v0 = ua ) ) " ()
  ((1324 nul 8)) -1 0)
 (1277 uniq "( move ul , ul ; move 4 , ul ; move ul - ua , ua * ua ) " () ()
  -1 0)
 (1278 equiv "( pen ua ; move 4 , ul ; ( v0 = ul ; v1 = 0 ) ) " () () 16 0)
 (1279 equiv "( pen ua ; pen ua ; ( move ua , 2 - 3 ; move 3 , 0 ) ) " () ()
  218 0)
 (1280 uniq "( move 4 , 0 - ul ; move ua + 2 , 0 - 2 ; move 0 , 2 ) " () ()
  -1 0)
 (1281 uniq "( ( pen ul ; move 4 , ua + 2 ) ; move ua , 4 ; pen 0 ) " () ()
  -1 0)
 (1282 equiv "( ( move ua , 4 ; pen ua ) ; move 0 , 3 * 2 ; pen ul ) " () ()
  277 0)
 (1283 uniq "( ( pen 2 ; pen 4 ) ; move ul , 0 ; move ua , 4 + ul ) " () ()
  -1 0)
 (1284 uniq "( pen ua + ua ; move ul , ua ; move ul - ua , ua - 4 ) " () ()
  -1 0)
 (1285 uniq "( pen ua - ul ; move 2 + ul , ua ; move ul + ul , ul ) " () ()
  -1 0)
 (1286 uniq "( move ua + ul , ua + ul ; pen 3 ; move ul + ul , ul ) " () ()
  -1 0)
 (1287 equiv "( ( move ua - ul , 0 ; v0 = ul ) ; move ua , v0 ) " () () 239
  0)
 (1288 uniq "( move ua - ul , 3 + ul ; pen ua ; move ul - ua , ul ) " () ()
  -1 0)
 (1289 equiv "( ( move ua , ua + ua ; v0 = 4 ) ; move ua , v0 ) " () () 33 0)
 (1290 uniq "( move ua + ul , 4 * 4 ; v0 = ul ; move ua , v0 ) " () () -1 0)
 (1291 uniq "( pen ua ; ( move 3 , 4 * 3 ; move ul , ul ) ; pen 0 ) " () ()
  -1 0)
 (1292 uniq "( ( pen 3 ; move ul , 4 + ul ; move 4 , ul ) ; pen 0 ) " () ()
  -1 0)
 (1293 uniq "( ( pen ul ; move 3 , 3 - 4 ; pen ua ) ; move ul , 2 ) " () ()
  -1 0)
 (1294 uniq "( move ua + ul , ua ; ( move ua , ua * ua ; pen ua ) ) " ()
  ((1465 nul 7)) -1 0)
 (1295 uniq "( ( move 4 , ul / 4 ; move 0 , 2 ) ; move ua + 4 , 0 ) " () ()
  -1 0)
 (1296 uniq "( pen 3 ; ( move ul , 2 ; move ua , ul ; v0 = ul ) ) " () () -1
  0)
 (1297 uniq "( ( move ul , ul ; move 3 , 4 ) ; pen 3 ; move ua , ul ) " () ()
  -1 0)
 (1298 uniq "( move 4 , 3 ; ( pen 4 ; move ua , ul ) ; move ul , ua ) " () ()
  -1 0)
 (1299 uniq "( move 2 , 3 ; ( move 3 , ul - 2 ; move ua , 4 / 2 ) ) " () ()
  -1 0)
 (1300 equiv "( move 4 , ul ; ( move ua , 4 ; v0 = ua ) ; pen ua ) " () () 44
  0)
 (1301 uniq "( move ul , ul ; pen ua ; ( v0 = 4 ; move ua , 4 ) ) " ()
  ((1532 nul 10)) -1 0)
 (1302 equiv "( ( pen ua ; v0 = ua ) ; pen ua ; move v0 , v0 ) " () () 14 0)
 (1303 uniq "( pen ua ; ( move ul , 2 ; move ul - ua , 2 ) ; pen 0 ) " () ()
  -1 0)
 (1304 equiv "( move ua , ul + ul ; v0 = ul ; move ua , v0 * 2 ) " () () 149
  0)
 (1305 uniq "( move 3 , ul ; pen ua + ul ; ( move ua , 4 ; pen 0 ) ) " () ()
  -1 0)
 (1306 equiv "( move 4 , ul ; ( pen 3 + ul ; move ua , 0 ) ; pen ua ) " () ()
  301 0)
 (1307 uniq "( v0 = ua + ul ; move ua , ul / ua ; move v0 , 3 ) " () () -1 0)
 (1308 equiv "( ( move ua , 0 ; pen ua ; move 4 , ul ) ; pen ua * 3 ) " () ()
  975 0)
 (1309 equiv "( pen 4 ; ( pen ul ; move ua , ua - ul ; move 4 , 0 ) ) " () ()
  2 0)
 (1310 uniq "( move 2 , ul ; ( pen 3 ; move ua , 4 - ua ) ; pen ul ) " () ()
  -1 0)
 (1311 uniq "( ( v0 = ua + 4 ; pen v0 ) ; move v0 , 3 / 4 ) " () () -1 0)
 (1312 uniq "( ( pen 3 ; pen ua ) ; move ul , 3 ; move 4 , ul - ua ) " () ()
  -1 0)
 (1313 uniq "( ( move ul - 3 , 3 - ua ; pen 3 ) ; move 3 + 2 , 4 ) " () () -1
  0)
 (1314 equiv "( ( move 4 , ul ; move ua , 4 ) ; pen ul ; pen 3 / ua ) " () ()
  44 0)
 (1315 uniq "( move 0 , ua * 4 ; move 3 * 3 , ua - 2 ; move ua , 2 ) " () ()
  -1 0)
 (1316 uniq "( ( move 0 , ua ; move 3 , 4 ) ; move ua + ul , 2 + 2 ) " () ()
  -1 0)
 (1317 uniq "( ( move 2 , 4 ; move ua , ua + 4 ) ; move ua , 0 - 2 ) " () ()
  -1 0)
 (1318 uniq "( v0 = 4 + ul ; ( move v0 , 2 ; move ua , v0 ) ) " () () -1 0)
 (1319 uniq "( move 2 , ul ; move 3 , ul - 2 ; move 3 + 3 , 4 / ua ) " () ()
  -1 0)
 (1320 uniq "( move 4 + 2 , 3 ; move ua - ul , 4 ; move 0 , ul / 4 ) " () ()
  -1 0)
 (1321 equiv "( move ua , ul - 4 ; ( v0 = 4 ; move v0 , v0 ) ) "
  ((743 del 4)) () 132 0)
 (1322 uniq "( ( pen ua ; move ua , 3 ) ; move ul , ul ; move ul , 4 ) " ()
  () -1 0)
 (1323 uniq "( ( pen 2 ; move ul , ua ) ; move ua , ul ; v0 = ul ) " () () -1
  0)
 (1324 equiv "( move 3 , ul ; ( pen 3 ; move ul , ul ) ; move ul , ua ) " ()
  () 1276 0)
 (1325 equiv "( v0 = 4 ; ( move 0 , ua ; pen ua ; move v0 , 0 ) ) "
  ((1337 sub 3)) () 12 0)
 (1326 equiv "( ( move ua , ua + ul ; pen 0 ) ; pen ua ; move ul , 0 ) " ()
  () 351 0)
 (1327 uniq "( move 4 , ul ; ( move 4 , ua + ul ; move 4 - 2 , 2 ) ) " () ()
  -1 0)
 (1328 equiv "( v0 = 0 ; ( pen 4 ; move ua , ul ) ; move v0 , 0 ) "
  ((637 del 5)) () 17 0)
 (1329 uniq "( move 3 , 4 + ul ; move ua / 3 , 2 / 4 ; move ua , 4 ) " () ()
  -1 0)
 (1330 equiv "( pen 3 ; ( pen ua + ul ; move ua , ua ) ; move 0 , ua ) " ()
  () 14 0)
 (1331 uniq "( pen 4 - ul ; move 2 , ua + 2 ; move 3 + 2 , ua - 4 ) " () ()
  -1 0)
 (1332 uniq "( pen 2 ; ( v0 = 4 ; move ua , ua ) ; move v0 , 2 ) " () () -1
  0)
 (1333 equiv "( ( move ua - ul , ul ; move ua , 2 ; pen 2 ) ; pen ua ) " ()
  () 196 0)
 (1334 equiv "( v0 = ua ; pen ua ; ( move 4 , 0 ; v1 = v0 ) ) " () () 12 0)
 (1335 equiv "( move 4 , ua + ul ; v0 = 0 ; move v0 + ua , v0 ) " () () 1 0)
 (1336 uniq "( ( move ua , ua ; pen ua ; pen ua ) ; move 4 , 4 + ul ) " () ()
  -1 0)
 (1337 uniq "( v0 = 4 ; ( move 4 , 3 ; pen ua ; move v0 , ul ) ) "
  ((1325 sub 3)) () -1 0)
 (1338 uniq "( pen ul ; move ua , ua / 4 ; ( move ua , 2 ; pen ul ) ) " () ()
  -1 0)
 (1339 equiv "( pen ul ; ( pen 3 + ul ; move ua , ul ; v0 = ua ) ) " () () 17
  0)
 (1340 equiv "( ( pen 2 + 2 ; move ua , 2 - ul ) ; v0 = ul - 4 ) " () () 17
  0)
 (1341 uniq "( move 4 , 2 ; ( pen 3 ; move 0 , 3 ) ; move 2 , 0 - 4 ) " () ()
  -1 0)
 (1342 uniq "( v0 = ua - ul ; move 4 , 0 ; move v0 , v0 / ua ) " () () -1 0)
 (1343 equiv "( move 4 , 0 ; move ua + ul , 0 - 2 ; v0 = ua * ua ) " () ()
  1226 0)
 (1344 equiv "( ( move ua , 0 ; v0 = ul - ua ) ; move v0 , v0 ) " () () 241
  0)
 (1345 uniq "( move 3 + ul , ua - ul ; move 2 * 2 , 2 ; move ua , 3 ) " () ()
  -1 0)
 (1346 equiv "( move 4 - ul , ua + 3 ; move 3 , ua ; move ua , 3 + 2 ) " ()
  () 1049 0)
 (1347 uniq "( move 4 , ul - 4 ; v0 = ua + ul ; move v0 , v0 ) " () () -1 0)
 (1348 uniq "( v0 = ul ; move 4 * 2 , ul ; move v0 - ua , v0 ) " () () -1 0)
 (1349 uniq "( move ul - ua , 0 ; move 2 * 2 , ul / 3 ; move 2 , ul ) " () ()
  -1 0)
 (1350 uniq "( move ua , 3 + 2 ; move 4 + ul , ul ; move ua , 4 - 2 ) " () ()
  -1 0)
 (1351 uniq "( move 2 , 2 - 3 ; move ua , ua + ul ; move 4 , ua - 4 ) " () ()
  -1 0)
 (1352 equiv "( move ua , ul ; ( v0 = ul ; move v0 + 3 , v0 ) ) " () () 67 0)
 (1353 uniq "( move ua + ul , 0 ; ( move 2 + ul , 4 ; move 2 , ul ) ) " () ()
  -1 0)
 (1354 uniq "( move ua , 3 ; ( move 3 , ua + 2 ; move ua , ua * 2 ) ) " () ()
  -1 0)
 (1355 uniq "( move ul , ua ; ( move ua , 2 ; move 4 * 2 , ua + 2 ) ) " () ()
  -1 0)
 (1356 uniq "( move ul - ua , 4 * 2 ; move ul , 4 ; move ua + 2 , 4 ) " () ()
  -1 0)
 (1357 uniq "( ( move ul , 0 ; move 0 , ul ) ; move ua + 3 , ul / 4 ) " () ()
  -1 0)
 (1358 uniq "( move 4 , 3 ; move 4 , ua / 2 ; move 0 - ua , ul / ua ) " () ()
  -1 0)
 (1359 uniq "( ( move 3 , ul ; move ua , 2 ) ; move ua / 3 , 2 - ua ) " () ()
  -1 0)
 (1360 uniq "( move 4 * 2 , ul ; ( move ul , 4 ; move ua + ul , 0 ) ) " () ()
  -1 0)
 (1361 uniq "( move 3 * 3 , ul ; move 4 , ua - ul ; v0 = ua - 2 ) " () () -1
  0)
 (1362 uniq "( ( move ul , 4 ; pen ua ) ; move ua , ul ; move ul , ul ) " ()
  () -1 0)
 (1363 uniq "( v0 = 0 ; ( move ua , v0 ; pen ua ) ; move ua , 2 ) " () () -1
  0)
 (1364 uniq "( move ua + ul , ul - 3 ; move 3 , 0 - 2 ; pen ul - 3 ) " () ()
  -1 0)
 (1365 uniq "( move 3 , ul ; ( v0 = ul - ua ; move ua + 3 , 4 ) ) " () () -1
  0)
 (1366 equiv "( pen 3 ; v0 = ul + ul ; move v0 * 3 , ua * v0 ) " () () 19 0)
 (1367 uniq "( move 3 + ul , 3 / ua ; move ua + ul , 2 ; pen 4 + 2 ) " () ()
  -1 0)
 (1368 uniq "( move 4 , ul - 4 ; ( move ua , ul ; move 4 + ul , 4 ) ) " () ()
  -1 0)
 (1369 uniq "( ( move ua , ul + ul ; pen 3 ) ; move 0 - ua , 2 - 3 ) " () ()
  -1 0)
 (1370 uniq "( move ua - ul , ul ; move 0 , 0 ; move ua - ul , 0 - 2 ) " ()
  () -1 0)
 (1371 uniq "( move ul , ul ; ( pen 3 ; v0 = 4 ) ; move ua , v0 ) " () () -1
  0)
 (1372 equiv "( pen ua + 3 ; ( move 3 + ul , ua ; v0 = ua - 2 ) ) " () () 12
  0)
 (1373 equiv "( v0 = ua + 2 ; move ua , ul ; move v0 , v0 - ua ) " () () 1091
  0)
 (1374 equiv "( ( pen ul ; v0 = ua ) ; pen v0 ; move v0 , v0 ) " () () 14 0)
 (1375 equiv "( ( v0 = ua ; pen ul ; pen v0 ) ; move v0 , v0 ) " () () 14 0)
 (1376 uniq "( ( move ua , ua * 3 ; pen 3 + ul ) ; move 4 , 2 - ua ) " () ()
  -1 0)
 (1377 uniq "( move ua , 0 - 4 ; move ua + ul , 4 ; move ul , 0 - ua ) " ()
  () -1 0)
 (1378 uniq "( v0 = 4 ; ( move 4 , 3 ; pen v0 ) ; move v0 , 0 ) " () () -1 0)
 (1379 uniq "( pen 3 ; ( move ul - ua , 3 ; move ul - ua , 4 * 2 ) ) " () ()
  -1 0)
 (1380 uniq "( move ua , ul ; ( move 2 - ua , 2 ; move ua + ul , 0 ) ) " ()
  () -1 0)
 (1381 uniq "( move 2 , ua ; move 4 , ua + ul ; move 4 + ul , ua + 3 ) " ()
  () -1 0)
 (1382 equiv "( move 2 , 0 ; ( move ua , ul ; move 3 + ul , ua + ua ) ) " ()
  () 397 0)
 (1383 uniq "( move ua + ul , ua ; move 3 + ul , 3 ; move ua , 3 - 4 ) " ()
  () -1 0)
 (1384 equiv "( v0 = ua * 4 ; move ua , v0 ; move v0 / ua , ul ) "
  ((746 del 4)) () 35 0)
 (1385 uniq "( move ua , ua ; ( move ua + ul , 0 - 2 ; v0 = ua ) ) " () () -1
  0)
 (1386 uniq "( move 2 , ua - 2 ; move 4 + ul , 4 ; move ul - ua , ul ) " ()
  () -1 0)
 (1387 uniq "( ( pen 2 ; move 4 , 2 + ul ; v0 = 2 ) ; move 2 , 4 ) " () () -1
  0)
 (1388 uniq "( move ua , 4 / ua ; move 3 , ul ; move ul - 3 , 0 - ul ) " ()
  () -1 0)
 (1389 uniq "( ( move ua , ua + 2 ; move 4 , ul ) ; move ua , ua + 3 ) " ()
  () -1 0)
 (1390 uniq "( move ua , 2 + ul ; move ua , 4 / 3 ; move 2 - ul , ul ) " ()
  () -1 0)
 (1391 uniq "( move 4 + ul , 3 ; move 3 - ul , ul ; move 3 + ul , ua ) " ()
  () -1 0)
 (1392 uniq "( move ul , 4 - ul ; move ua + 3 , 3 ; move ul - 3 , ul ) " ()
  () -1 0)
 (1393 uniq "( move ua + 3 , 2 / 3 ; move ul , ul - 4 ; move ul , ua ) " ()
  () -1 0)
 (1394 uniq "( move ua , ua ; move 2 / 4 , ua + ul ; move 3 , ua / 4 ) " ()
  () -1 0)
 (1395 equiv "( pen 0 ; move 0 , ul ; ( pen 4 ; move ua , ua ; pen 3 ) ) " ()
  () 17 0)
 (1396 uniq "( move 3 , ul ; move 4 , ul ; ( move 3 , 0 ; move 0 , 0 ) ) " ()
  () -1 0)
 (1397 equiv "( move ua , ua + ul ; ( move ua , 4 ; v0 = 2 + ul ) ) "
  ((566 del 6)) () 64 0)
 (1398 uniq "( ( move 4 , ul + ul ; move 0 , 3 / ua ) ; move ua , ul ) " ()
  () -1 0)
 (1399 uniq "( ( move ul + ul , ul / 3 ; move ua , ua ) ; move 3 , 2 ) " ()
  () -1 0)
 (1400 uniq "( move ua + ul , ul ; ( move ua , 3 ; move 4 / 3 , ua ) ) " ()
  () -1 0)
 (1401 uniq "( move ua + 2 , ua ; pen ua + 4 ; move 3 + ul , 3 + ul ) " () ()
  -1 0)
 (1402 uniq "( ( move 3 - ua , 4 ; move ua - ul , ul ) ; move 3 , ul ) " ()
  () -1 0)
 (1403 uniq "( pen ua ; move 0 - ul , ua / 3 ; move 3 + 2 , ua - ul ) " () ()
  -1 0)
 (1404 equiv "( v0 = ua + ul ; ( pen v0 ; move v0 + ul , ua ) ) "
  ((512 del 6)) () 69 0)
 (1405 uniq "( pen ua + 2 ; ( move ua + ul , 3 ; move ua , 2 - ua ) ) " () ()
  -1 0)
 (1406 uniq "( pen ua ; ( move 0 - ua , 3 ; move 2 + ul , ul + ul ) ) " () ()
  -1 0)
 (1407 uniq "( ( move ua , 2 ; pen ua ) ; v0 = ul ; move v0 , ul ) " () () -1
  0)
 (1408 uniq "( ( move ua , 0 - 2 ; pen ua ) ; move ul + ul , ua * 2 ) " () ()
  -1 0)
 (1409 uniq "( move ua , 0 ; pen 3 + 2 ; ( move 3 , ul ; move 0 , 0 ) ) " ()
  () -1 0)
 (1410 uniq "( move 4 , ua ; ( v0 = ua ; pen v0 ) ; move ul , ua ) " ()
  ((1504 nul 15)) -1 0)
 (1411 equiv "( ( move 3 + ul , 0 ; move ua - ul , ua - ul ) ; pen 3 ) " ()
  () 347 0)
 (1412 uniq "( move ul , 0 ; pen ua ; ( move 4 , 0 ; move 3 + 2 , 2 ) ) " ()
  () -1 0)
 (1413 uniq "( ( v0 = ul ; move 4 + 3 , 2 ) ; move ua , 2 / v0 ) " () () -1
  0)
 (1414 uniq "( pen 2 ; move 0 , 2 ; ( move 4 - ua , 2 ; move ua , 2 ) ) " ()
  () -1 0)
 (1415 uniq "( move ua , 4 ; ( pen 2 + ul ; move ua - 4 , ul + ul ) ) " () ()
  -1 0)
 (1416 uniq "( move ul + ul , 0 ; move 2 + ul , 3 ; move ua , ul + ul ) " ()
  () -1 0)
 (1417 equiv "( v0 = ua ; move v0 , v0 ; move v0 , v0 * v0 ) " () () 235 0)
 (1418 uniq "( move ua , ul / 4 ; move ul - ua , 2 / ua ; pen 4 * 3 ) " () ()
  -1 0)
 (1419 equiv "( move 4 + ul , ul ; move ua - ul , ua ; move 0 , 3 - ua ) " ()
  () 1 0)
 (1420 uniq "( move 2 + 2 , 2 ; pen 3 ; ( move 3 , 4 ; move ul , ua ) ) " ()
  () -1 0)
 (1421 uniq "( move ua - ul , ul ; move ul , ua + ul ; move 3 + 2 , 3 ) " ()
  () -1 0)
 (1422 equiv "( move ul + ul , ua ; move 2 + ul , ua ; move ua , 3 / 2 ) " ()
  () 1061 0)
 (1423 uniq "( move ul , ul ; move ua , 0 - 3 ; move ul - 4 , ua * ua ) " ()
  () -1 0)
 (1424 uniq "( move 4 , ua + 4 ; move ul , ua * ua ; move ua + 2 , ul ) " ()
  () -1 0)
 (1425 uniq "( ( move 4 , ua ; move ul , 2 ) ; pen 3 ; move 4 + 2 , 3 ) " ()
  () -1 0)
 (1426 equiv "( ( pen ul ; pen 3 ; v0 = ua - ul ) ; move ua , v0 ) " () () 83
  0)
 (1427 equiv "( move ua + ul , ul ; ( move 3 , ul ; move 0 , ua + ul ) ) " ()
  () 206 0)
 (1428 uniq "( move 2 + ul , ua * ua ; move ua , ul ; move 4 , ul - 3 ) " ()
  () -1 0)
 (1429 uniq "( move 0 , ul ; ( move ul - ua , 2 ; move ua , ua * ua ) ) " ()
  () -1 0)
 (1430 uniq "( move ul - ua , 0 ; ( move ul , ul - 3 ; move ua , ua ) ) " ()
  () -1 0)
 (1431 uniq "( move 2 , ua ; ( move ul - ua , ua ; move ua , 2 - ul ) ) " ()
  () -1 0)
 (1432 uniq "( ( move 4 , ul ; move ua , 2 ; pen 3 ) ; move 2 - 4 , 4 ) " ()
  () -1 0)
 (1433 uniq "( v0 = 3 - ul ; move ua + v0 , v0 ; move 4 , v0 ) " () () -1 0)
 (1434 uniq "( ( pen ul ; pen 3 ; move 0 , 4 + 2 ) ; move ua - 2 , 0 ) " ()
  () -1 0)
 (1435 uniq "( v0 = 3 ; move v0 + v0 , ul / v0 ; move ua , 4 ) " () () -1 0)
 (1436 uniq "( v0 = ua ; ( move v0 , 3 ; move ua , v0 * v0 ) ) " () () -1 0)
 (1437 uniq "( ( move ua - ul , ul ; move ua , 4 + ul ) ; v1 = ul ) " () ()
  -1 0)
 (1438 uniq "( pen ua + ul ; move ul + ul , ua * 4 ; move 4 - ua , 2 ) " ()
  () -1 0)
 (1439 uniq "( move ul / ua , ul ; ( move ua , 4 / ua ; move 4 , ul ) ) " ()
  () -1 0)
 (1440 uniq "( move 2 + ul , 3 ; pen ua ; ( move 3 , 4 ; pen 4 - 3 ) ) " ()
  () -1 0)
 (1441 equiv "( pen ua * ua ; move ua - ul , 4 - ua ; v0 = 2 / ua ) " () ()
  85 0)
 (1442 uniq "( move ua , ua + ua ; move ul , ul ; move 0 - ua , ua - 2 ) " ()
  () -1 0)
 (1443 uniq "( pen 3 ; ( move ua , 0 ; move ua , 3 - ul ; move 0 , 3 ) ) " ()
  () -1 0)
 (1444 equiv "( ( v0 = ua ; move v0 , 0 ; move 4 , ul ) ; pen v0 ) " () () 35
  0)
 (1445 uniq "( pen 2 ; move ul / ua , ua * ua ; move 4 + ul , 4 / ua ) " ()
  () -1 0)
 (1446 uniq "( v0 = 3 ; ( pen ua ; move ul , v0 ; move v0 , 3 ) ) " () () -1
  0)
 (1447 uniq "( ( pen 4 ; move 3 - ul , ua - ul ) ; move ua - ul , ul ) " ()
  () -1 0)
 (1448 equiv "( pen ua ; ( v0 = ua ; move 4 , v0 ; move v0 , 4 ) ) " () ()
  768 0)
 (1449 equiv "( move ua , 4 - ua ; move ul + ul , ul ; move ul + ul , 0 ) "
  () () 45 0)
 (1450 uniq "( pen 4 ; move ul , ua ; ( move 4 , 0 - 2 ; move ul , 2 ) ) " ()
  () -1 0)
 (1451 uniq "( move ul , 0 ; v0 = ua - ul ; move v0 + v0 , v0 ) "
  ((739 del 5)) () -1 0)
 (1452 equiv "( ( pen ua ; move 4 , ul ) ; move 0 , ul ; v0 = 0 - 4 ) " () ()
  16 0)
 (1453 uniq "( ( move ua , 0 ; move 4 + ul , 3 ) ; pen ua ; move 4 , 4 ) " ()
  () -1 0)
 (1454 uniq "( ( move ua , ul - 3 ; move 0 , 3 ) ; pen 3 ; move 4 , ul ) " ()
  () -1 0)
 (1455 equiv "( move ul / ua , 3 + ul ; move ua , ul ; move ua , ua + 3 ) "
  () () 799 0)
 (1456 uniq "( v0 = ua + ul ; move ua , v0 ; move 3 - v0 , v0 ) " () () -1 0)
 (1457 uniq "( ( move 2 / ua , 4 ; move ul , 2 ) ; pen ua ; move 3 , 2 ) " ()
  () -1 0)
 (1458 uniq "( move 4 + ul , 2 + ul ; move ul , ul ; move ul - ua , ul ) " ()
  () -1 0)
 (1459 uniq "( ( pen 3 * 2 ; move 4 , ua ) ; move ua , ul ; move 4 , 2 ) " ()
  () -1 0)
 (1460 uniq "( move 4 , 4 ; ( move 0 - ua , 3 ; pen 4 ) ; move ul , ul ) " ()
  () -1 0)
 (1461 uniq "( move ua + 2 , 0 ; ( pen ua ; move 3 , 3 - ul ; pen 0 ) ) " ()
  () -1 0)
 (1462 uniq "( move 4 , 4 ; ( move 4 / ua , 0 ; pen ua ; move 4 , ua ) ) " ()
  () -1 0)
 (1463 uniq "( move 4 , 0 ; ( pen 2 ; ( move ua , ul ; move ul , 4 ) ) ) " ()
  () -1 0)
 (1464 uniq "( pen 3 ; ( move 3 , 2 ; move ul , 2 ; v0 = ul / ua ) ) " () ()
  -1 0)
 (1465 equiv "( ( move ua + ul , ua ; move ua , ua * ua ) ; v0 = ua ) " () ()
  1294 0)
 (1466 uniq "( pen ua ; ( move 3 , ua ; move 3 + ul , 2 ; pen 0 - 3 ) ) " ()
  () -1 0)
 (1467 uniq "( ( move 4 , ul ; pen 4 ) ; move ul + ul , 4 ; move 4 , 4 ) " ()
  () -1 0)
 (1468 uniq "( pen ua ; ( move 2 , ul - 3 ; move ua , 4 - 3 ) ; pen 0 ) " ()
  () -1 0)
 (1469 uniq "( pen ul + ul ; move ua + ul , 3 + ul ; move ua , ua + 4 ) " ()
  () -1 0)
 (1470 uniq "( pen ua + ua ; ( move ua , 0 ; move ul + ul , ul + ul ) ) " ()
  () -1 0)
 (1471 equiv "( move ua , ua ; ( move ua , 2 ; v0 = 0 ) ; v1 = ua ) " () ()
  32 0)
 (1472 uniq "( move ua - 3 , 0 ; ( move 0 , ul ; pen ua ) ; move ua , 0 ) "
  () () -1 0)
 (1473 uniq "( ( move ul - ua , ua + ua ; pen ua ) ; move ul - ua , 2 ) " ()
  () -1 0)
 (1474 equiv "( ( move ua , ua ; move ul - ua , ua + ul ) ; pen 2 / ua ) " ()
  () 241 0)
 (1475 uniq "( ( ( pen 3 ; move ua , 4 + ul ) ; move ul , 4 ) ; pen 0 ) " ()
  () -1 0)
 (1476 uniq "( ( move 2 , ua + ul ; move ua , 2 ) ; pen 2 ; move 2 , ua ) "
  () () -1 0)
 (1477 uniq "( ( pen 2 ; move 0 , 3 ) ; move ua , ua ; move 2 , ul - ua ) "
  () () -1 0)
 (1478 uniq "( ( pen 3 ; move 0 , ua ; move ua , ua + 4 ) ; move 2 , ul ) "
  () () -1 0)
 (1479 equiv "( move ua , ua ; ( v0 = ul ; move 4 , v0 ) ; pen v0 ) " () ()
  35 0)
 (1480 uniq "( ( move 2 - ua , 2 ; move 0 , ul ) ; pen 3 ; move ul , ul ) "
  () () -1 0)
 (1481 uniq "( move 4 , ua ; ( move ua , ul - 4 ; move ua , 2 ) ; pen 0 ) "
  () () -1 0)
 (1482 uniq "( ( move 0 , ul ; pen 4 ; move ul , ua ) ; move 4 , 0 - ul ) "
  () () -1 0)
 (1483 uniq "( pen ua ; ( move 3 , 4 ; move ul , 3 ) ; move ua - ul , 0 ) "
  () () -1 0)
 (1484 uniq "( move 3 + ul , ul ; ( pen ua ; move 0 , 4 ) ; move 4 , ua ) "
  () () -1 0)
 (1485 uniq "( move ua , ul - ua ; move ul , 3 - ua ; move ua + ul , ul ) "
  ((629 del 6)) () -1 0)
 (1486 uniq "( move ua / 4 , ua ; move ua , ua ; move ua + ul , ul + ul ) "
  () () -1 0)
 (1487 uniq "( ( pen ua ; move ul , 2 ; move 3 / ua , 2 ) ; move 4 , ua ) "
  () () -1 0)
 (1488 uniq "( ( pen ua ; move 3 , ul ; move 2 , ul ) ; move ul / 2 , 3 ) "
  () () -1 0)
 (1489 uniq "( ( move ul , ul ; pen 4 ) ; move 3 , ul ; move 0 - 4 , ul ) "
  () () -1 0)
 (1490 uniq "( ( move 2 + ul , ul ; move 4 , ul ; pen 3 ) ; move ul , 3 ) "
  () () -1 0)
 (1491 equiv "( move ua + 2 , ul ; ( move ua , 4 ; v0 = 4 ) ; pen ua ) " ()
  () 163 0)
 (1492 uniq "( ( pen 4 ; move ua , 4 ; move ul , 2 ) ; move 3 , ul - ua ) "
  () () -1 0)
 (1493 equiv "( move 4 , ua + 2 ; ( move ua , ul ; pen 4 ; v0 = ua ) ) " ()
  () 40 0)
 (1494 uniq "( ( pen 2 ; move 4 , 0 ) ; move ul , ul ; move ul / 3 , ul ) "
  () () -1 0)
 (1495 uniq "( move ul , ul / 3 ; ( pen ua ; v0 = 0 ) ; move 3 , ul ) " () ()
  -1 0)
 (1496 uniq "( ( move ul , 2 ; move 4 , 4 ; pen ua ) ; move 2 / ua , ul ) "
  () () -1 0)
 (1497 uniq "( move ul , ua + 4 ; ( move 4 , 4 ; pen ua ; move ul , 4 ) ) "
  () () -1 0)
 (1498 uniq "( pen ua ; ( pen ua + 4 ; move 3 , 0 - ul ) ; move ua , 2 ) " ()
  () -1 0)
 (1499 uniq "( ( pen ua ; move 2 , ua + 3 ; move ua , ul / 2 ) ; pen 0 ) " ()
  () -1 0)
 (1500 equiv "( ( v0 = ua ; pen v0 ) ; move 4 , ua ; pen v0 * ua ) " () () 12
  0)
 (1501 uniq "( move 4 , ua * ua ; ( pen ua * 2 ; move 3 , ua ; pen 0 ) ) " ()
  () -1 0)
 (1502 uniq "( ( move ua , 3 ; move 2 , ul ; move ua , 2 ) ; move ul , ua ) "
  () () -1 0)
 (1503 uniq "( move ua , 4 ; ( pen ua + ul ; move 3 , 3 - ua ; pen 3 ) ) " ()
  () -1 0)
 (1504 equiv "( move 4 , ua * 2 ; pen 4 + ul ; ( move ul , 0 ; pen ul ) ) "
  () () 1410 0)
 (1505 uniq "( ( pen ua ; move ua + ul , ua ) ; move 0 , 0 ; move ua , 2 ) "
  () () -1 0)
 (1506 uniq "( move ua , ul ; pen 2 ; ( move 4 - ul , 4 + ul ; pen 2 ) ) " ()
  () -1 0)
 (1507 uniq "( ( move 4 , ua ; move ua , ul ) ; move ul , ul ; v0 = 0 ) " ()
  () -1 0)
 (1508 uniq "( ( move 4 , 3 * 3 ; pen ua + ul ; move 2 , ul ) ; pen ua ) " ()
  () -1 0)
 (1509 uniq "( move ul , ul ; ( move 3 , ul ; move ua , 4 ) ; move 2 , ul ) "
  () () -1 0)
 (1510 uniq "( ( move ua , 4 ; move ul , 3 ; move ua , 3 ) ; move ul , ul ) "
  () () -1 0)
 (1511 uniq "( move ua , ua ; ( move 0 - ul , ua ; pen ua ) ; move 2 , 2 ) "
  () () -1 0)
 (1512 uniq "( ( pen ua ; move 4 , ul ; pen 4 + ul ) ; move ul - 4 , 3 ) " ()
  () -1 0)
 (1513 equiv "( ( v0 = ua ; move ua , 2 ; v1 = ul ) ; move ua , ul ) " () ()
  62 0)
 (1514 equiv "( move 0 , ua - ul ; ( move ua , 0 ; pen ul ; move 4 , ua ) ) "
  () () 2 0)
 (1515 uniq "( ( move 3 , ul ; move 4 , ul ; move 4 , ul ) ; move ul , ul ) "
  () () -1 0)
 (1516 equiv "( pen ua ; ( move 0 , ua - ul ; move 3 , ul ; move ua , 2 ) ) "
  () () 526 0)
 (1517 uniq "( move 0 , ul ; ( move 3 , 3 ; pen ua ) ; move ua , ua + ul ) "
  () () -1 0)
 (1518 equiv "( ( move ua , ul ; move ua + 2 , 4 ; pen ul ) ; move 0 , ul ) "
  () () 161 0)
 (1519 uniq "( move 0 , ua ; move 2 * 2 , ul ; ( pen ua ; move ul , ul ) ) "
  () () -1 0)
 (1520 uniq "( ( move ua , 4 + ul ; move ul , 3 ) ; move ua , ul ; pen 0 ) "
  () () -1 0)
 (1521 uniq "( pen ua + ul ; ( pen 0 - ua ; pen 3 ; move ua , 4 / 3 ) ) " ()
  () -1 0)
 (1522 uniq "( ( move ua , 3 ; move ul , 0 ; pen ul ) ; move 4 , ua * ua ) "
  () () -1 0)
 (1523 equiv "( ( pen ua ; move 3 * 3 , ua * 4 ; pen 0 ) ; pen ul - ua ) " ()
  () 69 0)
 (1524 uniq "( pen ua ; ( move 3 , ul ; move ul , 3 - ul ; move ua , 3 ) ) "
  () () -1 0)
 (1525 uniq "( ( pen ul ; move 2 - ua , ul ; move 4 , ul ) ; move 4 , ul ) "
  () () -1 0)
 (1526 uniq "( move 2 , 0 ; pen ua + 2 ; ( move ul + ul , ul ; pen ua ) ) "
  () () -1 0)
 (1527 uniq "( ( pen 2 + ul ; move ul , 0 ) ; move ua + 2 , ul ; pen ul ) "
  () () -1 0)
 (1528 equiv "( pen ua * 2 ; ( pen ua ; move ua , ua - 3 ; move 0 , ul ) ) "
  () () 87 0)
 (1529 uniq "( ( move 3 + ul , ul - ua ; pen 0 ) ; pen ua ; move 3 , ua ) "
  () () -1 0)
 (1530 equiv "( pen ua ; ( ( move 4 , ul ; move ua , ua ) ; v0 = ul ) ) " ()
  () 77 0)
 (1531 equiv "( ( move ua , 0 ; move ua , 3 - ul ) ; pen 3 ; pen ul - ua ) "
  () () 32 0)
 (1532 equiv "( move ul , ul ; ( pen 0 ; pen 4 + ul ) ; move ua , ua + 4 ) "
  () () 1301 0)
 (1533 equiv "( ( pen ul / ua ; move 2 , 0 ) ; move 3 + ul , ul ; pen ua ) "
  () () 0 0)
 (1534 uniq "( ( pen 2 ; move ua , ul ; pen ua - 3 ) ; move ul - 3 , ul ) "
  () () -1 0)
 (1535 equiv "( move ua , ua ; move ul , ul ; ( move 3 , ua ; v0 = ua ) ) "
  () () 35 0)
 (1536 equiv "( ( pen ua ; move 4 * 2 , ua ; v0 = ua + ul ) ; pen ua ) " ()
  () 69 0)
 (1537 uniq "( pen 4 ; ( move ua + 2 , ua - 4 ; move ul , ul ; pen ul ) ) "
  () () -1 0)
 (1538 equiv "( ( pen 4 + ul ; move ul - ua , 2 ; pen ul ) ; v0 = ua ) " ()
  () 72 0)
 (1539 uniq "( move ul , ul + ul ; ( pen 4 ; pen ua / 2 ; move 4 , ul ) ) "
  () () -1 0)
 (1540 equiv
  "( ( move ul , 0 ; pen ua ; move ua , ul + ul ) ; move ul , 0 ) " () () 70
  0)
 (1541 equiv "( move ua , ul ; ( pen ua ; move ul , ua + ua ) ; v0 = 0 ) " ()
  () 351 0)
 (1542 uniq "( move ul , 0 ; ( move ua , ul ; pen ua + 3 ) ; move ul , ul ) "
  () () -1 0)
 (1543 uniq "( ( move ul , ul ; pen ua ; move 0 , ua ) ; move ua / 2 , ul ) "
  () () -1 0)
 (1544 uniq "( ( pen 2 ; move ua , ua - ul ; move ul , ua ) ; move 3 , ul ) "
  () () -1 0)
 (1545 uniq "( ( pen ua ; move 2 + ul , ul ; move ul , ul ) ; move 3 , ul ) "
  () () -1 0)
 (1546 uniq "( ( move ua , ul - 3 ; pen ua ) ; move ul , 3 ; move ua , ul ) "
  () () -1 0)
 (1547 uniq "( move ua , ua ; ( pen 4 ; move 3 , ua * ua ) ; move ul , ua ) "
  () () -1 0)
 (1548 uniq "( move ua , ul ; ( move 4 + ul , 2 ; move ul , ua ; pen ul ) ) "
  () () -1 0)
 (1549 equiv
  "( move ua , 3 ; ( move ul , 4 ; move ua - ul , ua ; pen ua ) ) " () () 39
  0)
 (1550 uniq "( ( move ua , ul ; move ul , ua ; pen ua ) ; move 3 - ul , 4 ) "
  () () -1 0)
 (1551 equiv "( ( pen 4 + ul ; move ul , ua ; move ua , ul ) ; v0 = ul ) " ()
  () 312 0)
 (1552 uniq "( pen 4 ; ( move 3 , ul ; move ul , ul ) ; move ul , ul - ua ) "
  () () -1 0)
 (1553 uniq "( ( move 4 , ul / 2 ; move ua , ul ; move ul , ul ) ; pen ul ) "
  () () -1 0)
 (1554 uniq "( pen ua * 2 ; ( move ua , ul ; move 2 , ua + ul ; pen ul ) ) "
  () () -1 0)
 (1555 equiv "( pen ua * 3 ; ( move ua , ul ; move ul , ul ) ; pen 0 - ul ) "
  () () 359 0)
 (1556 uniq "( move 2 - ua , ul ; pen ua ; ( move ul , ul + ul ; pen 3 ) ) "
  () () -1 0)
 (1557 equiv "( ( pen ul ; move ua + ul , 3 - ul ; move ua , 3 ) ; pen ul ) "
  () () 157 0)
 (1558 equiv "( ( pen ul / ua ; pen ua * 2 ; move ua , ua ) ; move ua , 4 ) "
  () () 276 0)
 (1559 equiv "( ( pen ul ; ( move ua , ul ; pen ua + ul ) ) ; move ul , 0 ) "
  () () 351 0)
 (1560 uniq "( move ua , ul ; pen ua ; ( move ua , ul + ul ; v0 = ul ) ) " ()
  () -1 0)
 (1561 equiv
  "( pen ul ; ( move ul + ul , ua + ul ; pen ua ) ; move ua , 2 ) " () () 471
  0)
 (1562 uniq "( pen ul ; ( pen ua ; move ua , ua - ul ) ; move ua , 3 - ul ) "
  () () -1 0)
 (1563 uniq "( pen ul ; ( move 2 , ul ; pen ul + ul ; move ul - ua , ul ) ) "
  () () -1 0))