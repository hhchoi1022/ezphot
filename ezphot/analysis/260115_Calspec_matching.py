#%%
from ezphot.utils import Tiles
#%%
tiles = Tiles()
# %%

raw_string=  '''109 VIR	A0III	 	3.73	-0.01	109vir	_mod_005	_stis_005	 
10 LAC	O9V	 	4.88	-0.21	10lac	_mod_006	_stis_008	<0.23
16 CYG B	G3V	 	6.20	0.66	16cygb	_mod_005	_stis_005	<0.21
18 SCO	G2V	 	5.50	0.65	18sco	_mod_006	_stis_006	 
1732526	A4V	 	12.53	0.12	1732526	_mod_007	_stisnic_009	1.40
17403462	A6V	 	12.48	0.20	1740346	 	_stisnic_005	 
1743045	A8III	 	13.52	0.28	1743045	_mod_007	_stisnic_009	<1.32
1757132	A3V	 	12.01	-0.10	1757132	_mod_007	_stiswfc_006	<0.37
1802271	A2V	 	11.53	0.02	1802271	_mod_007	_stiswfcnic_006	<0.82
1805292	A4V	 	12.28	0.14	1805292	_mod_006	_stisnic_008	<0.49
1808347	A3V	 	11.69	0.49	1808347	_mod_007	_stiswfc_006	1.65
1812095	A3V	 	12.01	-0.07	1812095	_mod_006	_stisnic_008	1.57
2M0036+18	L3.5	 	J=12.47	...	2m003618	 	_stiswfcnic_006	 
2M0559-14	T4.5	 	J=13.80	...	2m055914	 	_stiswfcnic_005	 
AGK+81 266	sdO	 	11.95	-0.36	agk_81d266	 	_stisnic_007*	 
ALPHA LYR	A0V	 	0.03	0.00	alpha_lyr	_mod_004	_stis_011*	 
BD+02 3375	A5	 	9.93	0.45	bd02d3375	_mod_005	_stis_008	 
BD-11 3759	M3.5V	 	11.32	1.60	bd11d3759	 	_stis_003	 
BD+17 47083	sdF8	 	9.47	0.44	bd_17d4708	 	_stisnic_007	 
BD+21 0607	F2	 	9.22	0.44	bd21d0607	_mod_004	_stis_007	 
BD+26 2606	A5	 	9.73	0.39	bd26d2606	_mod_005	_stis_008	 
BD+29 2091	F5	 	10.22	0.50	bd29d2091	_mod_004	_stis_007	 
BD+54 1216	sdF6	 	9.71	0.48	bd54d1216	_mod_003	_stis_006	 
BD+60 1753	A1V	 	9.65	0.07	bd60d1753	_mod_006	_stiswfc_005	<0.17
BD+75 325	sdO5	 	9.55	-0.33	bd_75d325	 	_stis_006*	 
C26202	F8IV	 	16.64	0.26	c26202	_mod_008	_stiswfcnic_007	 
DELTA UMI	A1V	 	4.34	0.03	delumi	_mod_005	_stis_005	0.04
ETA1 DOR	A0V	 	5.69	-0.01	eta1dor	_mod_005	_stis_005	0.18
ETA UMA	B3V	 	1.85	-0.10	etauma	_mod_005	_stis_008	 
FEIGE110	sdO	 	11.83	-0.30	feige110	 	_stisnic_008*	 
FEIGE34	sdO	 	11.14	-0.23	feige34	 	_stis_006*	 
G191B2B	DA.8	 	11.781	-0.33	g191b2b	_mod_012	_stiswfcnic_004*	 
GD153	DA1.2	 	13.349	-0.29	gd153	_mod_012	_stiswfcnic_004	 
GD71	DA1.5	 	13.032	-0.25	gd71	_mod_012	_stiswfcnic_004	 
GJ754.1A	DBQA5	 	12.29	0.05	gj7541a	_mod_001	_stis_004	 
GRW+70 5824	DA2.4	 	12.60	-0.06	grw_70d5824	_mod_001	_stiswfcnic_003	 
HD009051	G7III	 	8.92	0.81	hd009051	_mod_006	_stis_009	 
HD031128	F4V	 	9.14	0.41	hd031128	_mod_004	_stis_007	 
HD074000	sdF6	 	9.66	0.45	hd074000	_mod_004	_stis_007	 
HD101452	A2/3	 	8.20	-0.36	hd101452	_mod_005	_stis_005	<0.09
HD106252	G0	 	7.36	0.64	hd106252	_mod_007	_stis_009	 
HD11198011	F7V	 	8.38	0.53	hd111980	_mod_004	_stis_007	 
HD115169	G3V	 	9.20	0.69	hd115169	_mod_005	_stis_005	<0.11
HD116405	A0V	 	8.34	-0.07	hd116405	_mod_006	_stis_008	<0.10
HD128998	A1V	 	5.83	0.00	hd128998	_mod_004	_stis_004	0.05
HD142331	G5V	 	8.75	0.64	hd142331	_mod_006	_stis_006	 
HD14943	A5V	 	5.91	0.19	hd14943	_mod_004	_stis_006	0.05
HD158485	A4V	 	6.50	0.13	hd158485	_mod_005	_stis_007	0.42
HD159222	G1V	 	6.56	0.65	hd159222	_mod_007	_stis_009	 
HD160617	F	 	8.73	0.45	hd160617	_mod_003	_stis_006	 
HD163466	A6V	 	6.85	0.19	hd163466	_mod_005	_stis_007	0.17
HD1654595	A4V	 	6.86	0.13	hd165459	_mod_005	_stisnic_007	 
HD167060	G3V	 	8.92	0.64	hd167060	_mod_006	_stis_006	<0.09
HD172728	A0V	 	5.74	-0.05	hd172728	_mod_004	_stis_004	 
HD18060910	A0V	 	9.42	0.15	hd180609	_mod_006	_stis_008	<0.13
HD185975	G3V	 	8.10	0.68	hd185975	_mod_005	_stis_008	 
HD200654	G	 	9.11	0.63	hd200654	_mod_005	_stis_008	 
HD200775	B2Ve	 	7.43	0.33	hd200775	 	_stis_001	 
HD205905	G2V	 	6.74	0.62	hd205905	_mod_008	_stis_010	0.26
HD2094586	G0V	 	7.65	0.59	hd209458	_mod_003	_stisnic_008	 
HD2811	A3V	 	7.50	0.17	hd2811	_mod_006	_stis_006	 
HD37725	A3V	 	8.31	-0.19	hd37725	_mod_007	_stiswfc_006	0.13
HD37962	G2V	 	7.85	0.65	hd37962	_mod_009	_stis_011	<0.09
HD38949	G1V	 	7.80	0.57	hd38949	_mod_007	_stis_009	1.17
HD55677	A4V	 	9.41	0.06	hd55677	_mod_006	_stis_006	0.24
HD607537	B3IV	 	6.68	-0.09	hd60753	 	_stis_004	 
HD93521	O9Vp	 	6.99	-0.27	hd93521	_mod_003	_stis_007*	 
HS2027+0651	DO	 	16.9	...	hs2027	 	_stis_006	 
HZ4	DA3.4	 	14.51	0.09	hz4	_mod_001	_stis_009	 
HZ21	DO2	 	14.69	-0.33	hz21	 	_stis_007	 
HZ438	DA	 	12.91	-0.31	hz43	 	_stis_005*	 
HZ43B8	M3Ve	 	14.30	...	hz43b	 	_stis_005	 
HZ44	sdB	 	11.65	-0.23	hz44	 	_stis_006*	 
KF06T2	K1.5III	 	13.80	1.30	kf06t2	_mod_007	_stiswfcnic_006	 
KF08T3	K0.5III	 	13.18	1.21	kf08t3	_mod_006	_stisnic_005	 
KSI2 CETI	B9III	 	4.28	-0.04	ksi2ceti	_mod_006	_stis_008	 
LAMBDA LEP	B0.5V	 	4.29	-0.25	lamlep	_mod_006	_stis_008	0.35
LDS749B	DBQ4	 	14.674	-0.04	lds749b	_mod_007	_stisnic_008	 
MU COL	O9.5V	 	5.18	-0.28	mucol	_mod_006	_stis_008	0.12
NGC2506-G31	G1V	 	17.9	0.7	ngc2506g31	_mod_005	_stis_005	 
NGC6681-1	...	 	15.77	0.18	NGC6681_1	 	_stis_002	 
NGC6681-2	...	 	16.20	0.05	NGC6681_2	 	_stis_002	 
NGC6681-3	...	 	16.90	0.03	NGC6681_3	 	_stis_002	 
NGC6681-4	...	 	16.14	0.04	NGC6681_4	 	_stis_002	 
NGC6681-5	...	 	16.37	0.03	NGC6681_5	 	_stis_002	 
NGC6681-6	...	 	16.20	0.04	NGC6681_6	 	_stis_002	 
NGC6681-7	...	 	16.05	1.24	NGC6681_7	 	_stis_002	 
NGC6681-8	...	 	17.01	-0.01	NGC6681_8	 	_stis_002	 
NGC6681-9	...	 	15.80	0.18	NGC6681_9	 	_stis_002	 
NGC6681-10	...	 	19.14	-0.14	NGC6681_10	 	_stis_002	 
NGC6681-11	...	 	15.65	0.22	NGC6681_11	 	_stis_002	 
NGC6681-12	...	 	17.62	-0.08	NGC6681_12	 	_stis_002	 
P041C9	GOV	 	12.16	0.68	p041c	_mod_005	_stisnic_0010*	<0.44
P177D	G0V	 	13.49	0.60	p177d	_mod_006	_stisnic_011*	<1.00
P330E	G2V	 	12.92	0.64	p330e	_mod_008	_stiswfcnic_007*	<0.76
SDSS132811	DA	 	g=17.01	...	sdss132811	 	_stis_004	 
SF1615+001A	G	 	16.75	0.49	sf1615_001a	_mod_006	_stisnic_011	 
SIGMA ORI	O9.5V+B0.2V	 	3.79	-0.21	sigori	 	_stis_001	 
SIRIUS	A1V	 	-1.46	0.00	sirius	_mod_005	_stis_005*	 
SNAP-2	G0-5	 	16.23	0.86	snap2	_mod_007	_stiswfcnic_006	 
VB8	M7V	 	16.92	1.8	vb8	 	_stiswfcnic_004	 
WD0148+467	DA3.6	 	12.46	0.06	wd0148_467	_mod_001	_stiswfc_002	 
WD0227+050	DA2.5	 	12.80	-0.05	wd0227_050	_mod_001	_stiswfc_003	 
WD0809+177	DA	 	13.46	 	wd0809_177	_mod_001	_stiswfc_003	 
WD1105-048	DA3	 	13.06	0.04	wd1105_048	_mod_001	_stiswfc_003	 
WD1105-340	DA3.6	 	13.63	0.16	wd1105_340	_mod_001	_stiswfc_003	 
WD1202-232	DA5.7	 	12.90	0.14	wd1202_232	 	_stiswfc_002	 
WD1327-083	DA3.5	 	12.34	0.06	wd1327_083	_mod_001	_stiswfc_005	 
WD1544-377	DA4.8	 	12.80	0.30	wd1544_377	 	_stiswfc_002	 
WD1713+695	DA3.2	 	13.20	~0.2	wd1713_695	_mod_001	_stiswfc_003	 
WD1911+536	DA	 	13.26	~-0.1	wd1911_536	_mod_001	_stiswfc_002	 
WD1919+145	DA3.3	 	13.01	0.06	wd1919_145	_mod_001	_stiswfc_002	 
WD2039-682	DA3.0	 	13.30	0.05	wd2039_682	_mod_001	_stiswfc_002	 
WD2117+539	DA3.4	 	12.81	 	wd2117_539	_mod_001	_stiswfc_003	 
WD2126+734	DA3.1	 	12.83	-0.03	wd2126_734	_mod_001	_stiswfc_003	 
WD2149+021	DA2.8	 	12.74	-0.01	wd2149_021	_mod_001	_stiswfc_002	 
WD2341+322	DA3.8	 	12.94	0.12	wd2341_322	_mod_001	_stiswfc_005	 
WD0308-565	sdB	 	14.07	-0.11	wd0308_565	_mod_007	_stis_009	 
WD1057+719	DA1.2	 	14.68	...	wd1057_719	_mod_009	_stisnic_011	 
WD1657+343	DA.9	 	16.1	...	wd1657_343	_mod_009	_stiswfcnic_006	 
WDFS0122-30	DA	18.66	-	...	wdfs0122_30	_mod_001	_stis_001	 
WDFS0248+33	DA	18.52	-	...	wdfs0248_33	_mod_001	_stis_001	 
WDFS0458-56	DA	17.96	-	...	wdfs0458_56	_mod_001	_stis_001	 
WDFS0639-57	DA	18.37	-	...	wdfs0639_57	_mod_001	_stis_001	 
WDFS0956-38	DA	18.00	-	...	wdfs0956_38	_mod_001	_stis_001	 
WDFS1055-36	DA	18.20	-	...	wdfs1055_36	_mod_001	_stis_001	 
WDFS1110-17	DA	18.05	-	...	wdfs1110_17	_mod_001	_stis_001	 
WDFS1206-27	DA	16.67	-	...	wdfs1206_27	_mod_001	_stis_001	 
WDFS1214+45	DA	17.98	-	...	wdfs1214_45	_mod_001	_stis_001	 
WDFS1302+10	DA	17.24	-	...	wdfs1302_10	_mod_001	_stis_001	 
WDFS1434-28	DA	18.10	-	...	wdfs1434_28	_mod_001	_stis_001	 
WDFS1514+00	DA	15.88	-	...	wdfs1514_00	_mod_001	_stis_001	 
WDFS1535-77	DA	16.76	-	...	wdfs1535_77	_mod_001	_stis_001	 
WDFS1557+55	DA	17.69	-	...	wdfs1557_55	_mod_001	_stis_001	 
WDFS1814+78	DA	16.74	-	...	wdfs1814_78	_mod_001	_stis_001	 
WDFS1837-70	DA	17.91	-	...	wdfs1837_70	_mod_001	_stis_001	 
WDFS1930-52	DA	17.67	-	...	wdfs1930_52	_mod_001	_stis_001	 
WDFS2317-29	DA	18.53	-	...	wdfs2317_29	_mod_001	_stis_001	 
DFS2351+37	DA	18.23	-	...	wdfs2351_37	_mod_001	_stis_001	'''
# %%
star_names = []
sp_types = []
g_magnitudes = []
v_magnitudes = []
b_v_colors = []
names = []
models = []
stis_names = []
var_percentages = []
for line in raw_string.split('\n'):
    values = line.split('\t')
    star_names.append(values[0])
    sp_types.append(values[1])
    g_magnitudes.append(values[2])
    v_magnitudes.append(values[3])
    b_v_colors.append(values[4])
    names.append(values[5])
    models.append(values[6])
    stis_names.append(values[7])
    var_percentages.append(values[8])
# %%
raw_table2 = '''10 LAC	22 39 15.679	+39 03 00.97	-10	-0.32	-5.46	 	 
109 VIR	14 46 14.925	+01 53 34.38	-6.1	-114.03	-22.13	HD130109	 
16 CYG B	19 41 51.973	+50 31 03.09	-27.7	-134.79	-162.49	HD186427	 
18 SCO	16 15 37.270	-08 22 09.98	+11.9	232.16	-495.37	HD146233	 
1732526	17 32 52.630	+71 04 43.12	 	0.22	-2.71	2MASS J17325264+7104431	TYC 4424-1286-1
1740346	17 40 34.679	+65 27 14.77	 	-5.72	-3.44	2MASS J17403468+6527148	TYC 4207-219-1
1743045	17 43 04.486	+66 55 01.66	 	1.10	-2.79	2MASS J17430448+6655015	 
1757132	17 57 13.233	+67 03 40.77	 	0.41	-14.03	2MASS J17571324+6703409	TYC 4212-455-1
1802271	18 02 27.163	+60 43 35.54	 	5.40	2.17	2MASS J18022716+6043356	TYC 4201-1542-12
1805292	18 05 29.275	+64 27 52.13	 	-1.64	10.06	2MASS J18052927+6427520	TYC 4209-1396-1
1808347	18 08 34.736	+69 27 28.72	 	4.43	8.52	2MASS J18083474+6927286	TYC 4433-1800-1
1812095	18 12 09.567	+63 29 42.26	 	4.07	1.31	2MASS J18120957+6329423	TYC 4205-1677-1
1812524	18 12 52.381	+60 02 31.95	 	-3.03	-8.02	2MASS J18125240+6002319	TYC 4201-1717-1
2M0036+18	00 36 16.112	+18 21 10.29	+19	901.56	124.02	2MASS J00361617+1821104	 
2M0559-14	05 59 19.188	-14 04 49.22	 	570.20	-337.59	2MASS J05591914-1404488	 
AGK+81 266	09 21 19.177	+81 43 27.63	34.11	-11.26	-51.26	 	 
ALPHA LYR	18 36 56.336	+38 47 01.28	-21	200.94	286.23	Vega	HD172167
BD+02 3375	17 39 45.595	+02 24 59.61	-398	-366.01	75.12	 	 
BD-11 3759	14 34 16.812	-12 31 10.42	-1	-355.04	593.22	 	 
BD+17 4708	22 11 31.375	+18 05 34.16	-291	506.37	60.49	 	 
BD+21 0607	04 14 35.516	+22 21 04.25	+340	425.99	-301.87	HD284248	 
BD+26 2606	14 49 02.355	+25 42 09.14	+33	-5.88	-347.60	 	 
BD+28 4211	21 51 11.022	+28 51 50.37	 	-34.73	-56.85	WD2148+286	 
BD+29 2091	10 47 23.163	+28 23 55.93	+83	177.50	-824.83	 	 
BD+54 1216	08 19 22.572	+54 05 09.63	+66	-34.20	-628.56	HD233511	 
BD+60 1753	17 24 52.275	+60 25 50.75	-27.3	4.89	3.76	 	 
BD+75 325	08 10 49.490	+74 57 57.94	-50	7.17	10.30	 	 
C26202	03 32 32.843	-27 51 48.58	 	 	 	2MASS J03323287-2751483	[B2010] C26202
DELTA UMI	17 32 12.997	+86 35 11.26	-7.6	10.17	53.97	HD166205	 
ETA UMA	13 47 32.438	+49 18 47.76	-13	-121.17	-14.91	HD120315	 
ETA1 DOR	06 06 09.382	-66 02 22.63	+17.6	13.66	27.82	HD42525	 
FEIGE110	23 19 58.400	-05 09 56.17	 	-10.68	0.31	 	 
FEIGE34	10 39 36.738	+43 06 09.21	11.01	12.54	-25.41	WD1036+433	 
G191B2B	05 05 30.618	+52 49 51.92	+22.1	12.70	-93.42	BD+52 913	EGGR 247 WD0501+527
GD153	12 57 02.322	+22 01 52.63	+8.3	-38.40	-202.99	WD1254+223	 
GD71	05 52 27.620	+15 53 13.23	+30	76.73	-172.96	WD0549+158	 
GJ754.1A	19 20 34.923	-07 40 00.07	 	-61.28	-161.77	LAWD74	WD1917-077
GRW+70 5824	13 38 50.478	+70 17 07.64	+26	-402.08	-24.56	LAWD52	WD1337+705
HD2811	00 31 18.490	-43 36 23.00	 	-6.02	-4.18	 	 
HD009051	01 28 46.503	-24 20 25.44	-72	53.56	-17.03	 	 
HD14943	02 22 54.675	-51 05 31.66	+5	22.33	66.38	HR 701	 
HD031128	04 52 09.910	-27 03 50.94	+112	164.76	-26.52	 	 
HD37725	05 41 54.370	+29 17 50.96	 	15.05	-26.93	 	 
HD37962	05 40 51.966	-31 21 03.99	+3	-59.65	-365.23	 	 
HD38949	05 48 20.059	-24 27 49.85	+3	-30.44	-35.42	 	 
HD55677	07 14 31.290	+13 51 36.79	-2	-2.66	-6.81	 	 
HD60753	07 33 27.319	-50 35 03.31	+20	-3.12	5.31	 	 
HD074000	08 40 50.804	-16 20 42.51	+206	350.82	-484.16	 	 
HD93521	10 48 23.512	+37 34 13.09	-14	0.22	1.72	 	 
HD101452	11 40 13.651	-39 08 47.67	 	-34.17	-20.98	 	 
HD106252	12 13 29.510	+10 02 29.89	+16	22.86	-280.01	 	 
HD111980	12 53 15.053	-18 31 20.01	+155	299.49	-796.09	 	 
HD115169	13 15 47.388	-29 30 21.18	+21.2	-110.57	-82.09	 	 
HD116405	13 22 45.124	+44 42 53.91	-19	8.01	-10.29	 	 
HD128998	14 38 15.222	+54 01 24.02	-3.0	17.28	-18.99	HR 5467	 
HD142331	15 54 19.788	-08 34 49.37	-70.8	-105.98	-23.73	 	 
HD158485	17 26 04.837	+58 39 06.83	-30	-9.10	14.67	HR 6514	 
HD159222	17 32 00.992	+34 16 16.13	-52	-240.70	63.71	HR 6538	 
HD160617	17 42 49.324	-40 19 15.51	+100	-62.39	-393.23	 	 
HD163466	17 52 25.376	+60 23 46.94	-16	-2.73	42.67	 	 
HD165459	18 02 30.741	+58 37 38.16	-19.2	-13.06	24.61	 	 
HD167060	18 17 44.143	-61 42 31.62	+15.2	88.52	-145.15	 	 
HD172728	18 37 33.517	+62 31 35.67	-10.5	-8.98	49.58	 	 
HD180609	19 12 47.200	+64 10 37.17	 	-3.06	-7.79	 	 
HD185975	20 28 18.740	-87 28 19.94	-19	169.76	-56.99	 	 
HD200654	21 06 34.751	-49 57 50.28	-45	193.94	-273.89	 	 
HD200775	21 01 36.921	+68 09 47.79	-3.4	7.60	-2.82	 	 
HD205905	21 39 10.151	-27 18 23.67	-17	384.10	-83.96	 	 
HD209458	22 03 10.773	+18 53 03.55	-15	29.58	-17.89	 	 
HS2027+0651	20 29 32.506	+07 01 07.70	 	9.24	1.53	WD2027+0651	 
HZ4	03 55 21.988	+09 47 18.13	+82.1	173.27	-5.57	WD0352+096	 
HZ21	12 13 56.264	+32 56 31.36	 	-100.88	30.13	WD1211+332	 
HZ43	13 16 21.853	+29 05 55.38	+54	-157.96	-110.23	WD1314+293	 
HZ43B	13 16 21.495	+29 05 53.07	 	 	 	 	 
HZ44	13 23 35.263	+36 07 59.55	 	-66.27	-4.52	WD1321+364	 
KF01T5	18 04 03.894	+66 55 43.81	-22	-0.40	0.47	2MASS J18040388+6655437	[RMC2005] KF01T5
KF06T1	17 57 58.486	+66 52 29.41	-41	-2.13	-8.70	2MASS J17575849+6652293	[RMC2005] KF06T1
KF06T2	17 58 37.995	+66 46 52.11	 	0.62	-4.42	2MASS J17583798+6646522	[RMC2005] KF06T2
KF08T3	17 55 16.216	+66 10 11.61	-50	2.09	-6.39	2MASS J17551622+6610116	[RMC2005] KF08T3
KSI2 CETI	02 28 09.557	+08 27 36.22	+12	23.71	-4.79	 	 
LAMBDA LEP	05 19 34.524	-13 10 36.44	+20	-3.30	-4.91	 	 
LDS749B	21 32 16.233	+00 15 14.40	-81	413.23	27.27	LAWD87	WD2129+000
MU COL	05 45 59.895	-32 18 23.16	+109	2.99	-22.03	 	 
NGC2506G31	08 00 14.212	-10 47 29.47	 	 	 	 	 
NGC6681-1	18 43 13.319	-32 17 25.36	218.7	 	 	 	 
NGC6681-2	18 43 13.165	-32 17 25.82	218.7	 	 	 	 
NGC6681-3	18 43 13.041	-32 17 25.41	218.7	 	 	 	 
NGC6681-4	18 43 12.893	-32 17 26.43	218.7	 	 	 	 
NGC6681-5	18 43 12.872	-32 17 26.56	218.7	 	 	 	 
NGC6681-6	18 43 12.755	-32 17 25.69	218.7	 	 	 	 
NGC6681-7	18 43 12.639	-32 17 27.09	218.7	 	 	 	 
NGC6681-8	18 43 12.278	-32 17 27.51	218.7	 	 	 	 
NGC6681-9	18 43 12.199	-32 17 27.11	218.7	 	 	 	 
NGC6681-10	18 43 12.146	-32 17 26.58	218.7	 	 	 	 
NGC6681-11	18 43 12.049	-32 17 27.52	218.7	 	 	 	 
NGC6681-12	18 43 11.902	-32 17 27.91	218.7	 	 	 	 
P041C	14 51 57.980	+71 43 17.39	-22	-49.32	19.58	2MASS J14515797+7143173	GSPC P 41-C
P177D	15 59 13.579	+47 36 41.91	 	-7.90	1.57	2MASS J15591357+4736419	GSPC P177-D
P330E	16 31 33.813	+30 08 46.40	-53	-8.99	-38.77	2MASS J16313382+3008465	GSPC P330-E
SDSS132811	13 28 11.498	+46 30 50.94	 	-130.86	-30.80	SDSS J132811.45+463050.8	WD1326+467
SF1615+001A	16 18 14.240	+00 00 08.61	 	2.40	-10.94	2MASS J16181422+0000086	[B2010] SF1615+001A
SIGMA ORI	05 38 44.765	-02 36 00.28	+29.9	4.6	-0.4	 	 
SIRIUS	06 45 08.917	-16 42 58.02	-6	-546.01	-1223.1	 	 
SNAP-1	16 29 35.747	+52 55 53.61	 	-3.16	-20.80	2MASS J16293576+5255532	 
SNAP-2	16 19 46.103	+55 34 17.86	 	-2.91	-10.95	2MASS J16194609+5534178	 
VB8	16 55 35.256	-08 23 40.75	+15	-813.42	-870.61	 	 
WD0148+467	01 52 02.962	+47 00 06.65	+64.0	4.64	122.02	 	 
WD0227+050	02 30 16.628	+05 15 50.70	+16.5	76.96	-24.50	 	 
WD0809+177	08 12 37.809	+17 37 01.43	 	73.48	-87.17	 	 
WD1105-048	11 07 59.950	-05 09 26.03	+47.9	-55.55	-442.63	 	 
WD1105-340	11 07 47.897	-34 20 51.49	 	39.89	-263.43	 	 
WD1202-232	12 05 26.674	-23 33 12.14	+23.3	41.82	226.56	 	 
WD1327-083	13 30 13.637	-08 34 29.47	+36	-1111.1	-472.38	Gaia DR2 3630035787972473600	 
WD1544-377	15 47 30.021	-37 55 08.46	+21.1	-423.69	-209.11	 	 
WD1713+695	17 13 06.091	+69 31 25.51	 	-55.51	-343.04	 	 
WD1911+536	19 12 48.566	+53 43 13.45	 	144.45	136.06	 	 
WD1919+145	19 21 40.418	+14 40 41.40	+49.5	-33.02	-75.93	 	 
WD2039-682	20 44 21.459	-68 05 21.36	+57.0	182.10	-228.17	 	 
WD2117+539	21 18 56.264	+54 12 41.24	 	-85.45	193.19	 	 
WD2126+734	21 26 57.656	+73 38 44.67	8.0	55.34	-314.20	 	 
WD2149+021	21 52 25.379	+02 23 19.58	+28.2	15.32	-300.53	 	 
WD2341+322	23 43 50.721	+32 32 46.73	-15.7	-215.82	-59.74	Gaia DR2 2871730307948650368	LAWD93
WD0308-565	03 09 47.918	-56 23 49.41	-68	149.24	66.92	 	 
WD0320-539	03 22 14.820	-53 45 16.47	+57.8	6.56	-59.93	 	 
WD0947+857	09 57 54.296	+85 29 40.88	 	-28.13	-27.27	 	 
WD1026+453	10 29 45.295	+45 07 04.93	 	-90.46	1.68	 	 
WD1057+719	11 00 34.243	+71 38 02.92	+76	-43.64	-21.75	 	 
WD1657+343	16 58 51.113	+34 18 53.32	 	8.77	-31.23	Gaia DR2 1337946019956816256	 
WDFS0122-30	01 22 00.700	-30 52 03.75	 	20.62	-12.30	Gaia DR2 5028544686500198144	2QZ J012200.6-305203
WDFS0248+33	02 48 54.960	+33 45 48.32	+47.4	4.09	-4.76	SDSS J024854.96+334548.3	 
WDFS0458-56	04 58 22.855	-56 37 34.50	 	143.60	66.49	Gaia DR3 4764189621230467584	 
WDFS0639-57	06 39 41.434	-57 12 31.86	 	17.51	43.58	Gaia DR3 5484605140287436416	 
WDFS0956-38	09 56 57.020	-38 41 29.53	 	-8.27	-46.08	Gaia DR3 5421579652019276160	 
WDFS1055-36	10 55 25.384	-36 12 15.47	 	-21.35	46.13	Gaia DR3 5401230062610609920	 
WDFS1110-17	11 10 59.429	-17 09 54.18	+83.9	5.45	-8.02	SDSS J111059.44-170954.2	 
WDFS1206-27	12 06 20.351	-27 29 40.68	 	3.02	2.80	WD1203-272	 
WDFS1214+45	12 14 05.111	+45 38 18.40	 	0.28	13.92	WD1211+459	 
WDFS1302+10	13 02 34.436	+10 12 38.99	 	-12.86	-16.84	WD1300+104	 
WDFS1434-28	14 34 59.587	-28 19 03.59	 	-48.56	18.60	Gaia DR3 6222123588482712832	 
WDFS1514+00	15 14 21.273	+00 47 52.81	+15.6	4.35	-26.86	WD1511+009	SDSSJ151421.27+004752.8
WDFS1535-77	15 35 45.310	-77 24 44.13	 	-26.88	-43.75	WD1529-772	 
WDFS1557+55	15 57 45.402	+55 46 09.70	 	-11.68	-21.48	WD1556+559	 
WDFS1814+78	18 14 24.138	+78 54 02.90	 	-10.74	11.54	WD1817+788	 
WDFS1837-70	18 37 17.874	-70 02 51.30	 	10.38	-75.99	Gaia DR3 6431766714636858240	 
WDFS1930-52	19 30 18.957	-52 03 46.02	 	21.55	-33.29	Gaia DR3 6646236009641999488	 
WDFS2317-29	23 17 20.289	-29 03 22.05	 	3.99	25.05	WD2314-293	 
WDFS2351+37	23 51 44.297	+37 55 42.73	+78.5	-16.41	-9.94	SDSS J235144.29+375542.6	 '''
# %%
star_names_tbl2 = []
ra = []
dec = []
vr = []
pm_ra = []
pm_dec = []
simbad_name = []
alt_simbad_name = []
for line in raw_table2.split('\n'):
    values = line.split('\t')
    star_names_tbl2.append(values[0])
    ra.append(values[1])
    dec.append(values[2])
    vr.append(values[3])
    pm_ra.append(values[4])
    pm_dec.append(values[5])
    simbad_name.append(values[6])
    alt_simbad_name.append(values[7])
# %%
from astropy.table import Table

tbl1 = Table()
tbl1['star_names'] = star_names
tbl1['sp_types'] = sp_types
tbl1['g_magnitudes'] = g_magnitudes
tbl1['v_magnitudes'] = v_magnitudes
tbl1['b_v_colors'] = b_v_colors
tbl1['names'] = names
tbl1['models'] = models
tbl1['stis_names'] = stis_names
tbl1['var_percentages'] = var_percentages
# %%
tbl2 = Table()
tbl2['star_names'] = star_names_tbl2
tbl2['ra'] = ra
tbl2['dec'] = dec
tbl2['vr'] = vr
tbl2['pm_ra'] = pm_ra
tbl2['pm_dec'] = pm_dec
tbl2['simbad_name'] = simbad_name
tbl2['alt_simbad_name'] = alt_simbad_name
# %%
from astropy.table import join
merged_tbl =  join(tbl1, tbl2, keys = 'star_names',join_type = 'outer')
# %%
import numpy as np

def to_float(x):
    try:
        return float(x)
    except:
        return np.nan
# magnitude cut
magnitudes = merged_tbl['v_magnitudes']
magnitudes = np.array([to_float(x) for x in magnitudes])
magnitudes_mask = (np.isfinite(magnitudes)) & (magnitudes > 12) & (magnitudes < 18)
merged_tbl = merged_tbl[magnitudes_mask]
#%%
# declination cut
from astropy.coordinates import SkyCoord
merged_tbl = merged_tbl[(merged_tbl['ra'].mask == False) & (merged_tbl['dec'].mask == False)]
from astropy.coordinates import SkyCoord
import astropy.units as u

coords = SkyCoord(
    merged_tbl['ra'],
    merged_tbl['dec'],
    unit=(u.hourangle, u.deg),
    frame='icrs'
)
coords_mask = coords.dec.value < 20
merged_tbl = merged_tbl[coords_mask]
# %%
merged_tbl
list_ra = coords[coords_mask].ra.value
list_dec = coords[coords_mask].dec.value

# %%
tiles = Tiles()
# %%
all_tiles = []
for ra, dec in zip(list_ra, list_dec):
    tile = tiles.find_overlapping_tiles([ra], [dec])[0]
    all_tiles.append(tile)
#%%
all_tile_str = [tile['id'][0] for tile in all_tiles]
all_tile_ra = [tile['ra'][0] for tile in all_tiles]
all_tile_dec = [tile['dec'][0] for tile in all_tiles]
#%%
merged_tbl['tileid'] = all_tile_str
merged_tbl['tile_ra'] = all_tile_ra
merged_tbl['tile_dec'] = all_tile_dec
#%%
# %%
for row in merged_tbl:
    print(row['star_names'])
# %%
coords[coords_mask].galactic
# %%
merged_tbl.write('./calspec.sacii_fixed_width', format = 'ascii.fixed_width', overwrite = True)
# %% DOWNLOAD SPECTTRUM FILES 
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
OUTDIR = "calspec"

os.makedirs(OUTDIR, exist_ok=True)

def get_file_links(url):
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".fits"):   # CALSPEC spectra are FITS
            links.append(urljoin(url, href))
    return links


def download_file(url, outdir):
    filename = os.path.join(outdir, os.path.basename(url))
    if os.path.exists(filename):
        return  # skip existing

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(filename, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=os.path.basename(url)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


if __name__ == "__main__":
    print("Fetching file list...")
    files = get_file_links(BASE_URL)
    print(f"Found {len(files)} CALSPEC files")

    for url in files:
        download_file(url, OUTDIR)

    print("Download complete.")

# %%
