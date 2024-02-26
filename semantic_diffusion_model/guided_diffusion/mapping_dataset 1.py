from operator import indexOf
import os
from typing import Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import mapping_utils_new as mapping_utils
import pandas as pd
import glob
#from tabulate import tabulate
from tqdm import tqdm
from PIL import Image
from termcolor import colored

try:
    from solo.data.pretrain_dataloader import FullTransformPipeline
except ModuleNotFoundError as e:
    print(__file__, ":", e)

DATASET_MEAN = 243.0248
DATASET_STD = 391.5486

DIAGNOSIS_DATA = "/home1/ssl-phd/data/Mapping_data_diagnosis_raw.csv"

# Test and validation samples are selected on a patient level (i.e. all data of a patient is either training, test, or validation)
# Test and validation patients were selected randomly
# from the list of patients without missing contours, or incorrect mapping files
# fmt: off
# fmt: off prevents black from autoformatting these lines
TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = [3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]
TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]
INTEROBSERVER_PATIENTS = range(223, 262+1)

# Adding HCM patients
HCM_TRAIN = [282, 326, 359, 327, 279, 302, 285, 308, 301, 333, 361, 269, 317, 305, 311, 266, 272, 312, 342, 340, 334, 339, 357, 356, 335, 264, 324, 293, 291, 306, 347, 341, 363, 268, 292, 319, 332, 349, 267, 360, 320, 343, 358, 309, 330, 278, 284, 300, 350, 277, 316, 345, 299, 313, 352, 353, 295, 263, 276, 294, 270, 281, 325, 364, 321, 315, 273, 288, 362, 314, 310, 329, 355, 322, 337, 338, 303, 351, 296, 283, 323, 336]
HCM_VAL = [280, 286, 344, 287, 298, 328, 354, 348, 304, 365]
HCM_TEST = [274, 271, 289, 265, 307, 318, 297, 346, 275, 331, 290]

ATHLETE_TRAIN = [366, 367, 368, 369, 370, 371, 372, 375, 376, 377, 380, 382, 383, 384, 385, 386, 387, 388, 389, 392, 393, 394, 395, 396, 397, 398, 400, 401, 402, 404, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 422, 423, 424, 427, 428, 429, 430, 431, 433, 435, 436, 437, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448]
ATHLETE_VAL = [373, 374, 379, 390, 399, 425, 426, 438]
ATHLETE_TEST = [378, 381, 391, 403, 405, 410, 421, 432, 434]

AMYLOIDOSIS_TRAIN = [452, 454, 456, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 470, 471, 472, 473, 475, 476, 477, 478]
AMYLOIDOSIS_VAL = [453, 455, 457, 469, 474, 479, 480]

UNLABELED = list(range(481, 1262)) # patients without any label (neither segmentation nor diagnosis)

AKUT_MYOCARDITIS_TRAIN = [
    1262, 1264, 1265, 1266, 1268, 1270, 1271, 1272, 1274, 1275, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1300, 1301, 1303, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1441, 1487, 1488, 1489, 1491, 1492, 1494, 1496, 1498, 1499, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1534, 1535, 1536, 1537, 1538, 1540, 1541, 1542, 1546, 1554, 1555, 1556] # num = 96
AKUT_MYOCARDITIS_VAL = [
    1267, 1277, 1304, 1445, 1490, 1493, 1497, 1500, 1519, 1533, 1539, 1544] # num = 12
AKUT_MYOCARDITIS_TEST = [
    1263, 1269, 1273, 1276, 1299, 1302, 1442, 1495, 1518, 1543, 1545, 1553] # num = 12

LEZAJLOTT_MYOCARDITIS_TRAIN = [
    1315, 1318, 1319, 1322, 1323, 1326, 1327, 1329, 1331, 1332, 1333, 1335, 1336, 1337, 1338, 1440, 1443, 1444, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1461, 1463, 1464, 1465, 1466, 1467, 1468, 1470, 1471, 1472, 1474, 1477, 1478, 1482, 1483, 1484, 1485, 1486, 1548, 1550, 1551, 1552] # num = 52
LEZAJLOTT_MYOCARDITIS_VAL = [
    1316, 1317, 1320, 1321, 1334, 1339, 1459, 1469, 1475, 1480, 1481] # num = 11
LEZAJLOTT_MYOCARDITIS_TEST = [
    1324, 1325, 1328, 1330, 1446, 1460, 1462, 1473, 1476, 1479, 1549] # num = 11

DIAG_SEG_23_08_TRAIN = [1584, 1731, 1712, 1639, 1714, 1625, 1758, 1612, 1661, 1700, 1668, 1788, 1784, 1793, 1649, 1735, 1654, 1603, 1671, 1613, 1657, 1568, 1705, 1642, 1767, 1794, 1771, 1592, 1707, 1647, 1804, 1739, 1650, 1799, 1640, 1748, 1616, 1696, 1627, 1695, 1681, 1609, 1738, 1744, 1672, 1809, 1567, 1734, 1798, 1750, 1772, 1646, 1608, 1765, 1723, 1756, 1593, 1641, 1716, 1770, 1620, 1599, 1789, 1774, 1753, 1763, 1591, 1638, 1601, 1619, 1733, 1644, 1787, 1699, 1722, 1812, 1569, 1571, 1682, 1736, 1785, 1648, 1690, 1803, 1651, 1791, 1691, 1576, 1718, 1615, 1807, 1623, 1725, 1667, 1602, 1761, 1796, 1617, 1587, 1800, 1562, 1678, 1780, 1776, 1610, 1680, 1595, 1709, 1762, 1683, 1577, 1582, 1653, 1792, 1583, 1692, 1754, 1730, 1594, 1811, 1775, 1559, 1560, 1614, 1656, 1708, 1808, 1740, 1563, 1726, 1635, 1677, 1564, 1636, 1589, 1628, 1729, 1660, 1757, 1766, 1674, 1605, 1606, 1783, 1732, 1751, 1578, 1622, 1694, 1795, 1659, 1728, 1769, 1773, 1720, 1581, 1621, 1645, 1721, 1611, 1643, 1713, 1777, 1597, 1669, 1565, 1684, 1666, 1598, 1588, 1633, 1790, 1737, 1686, 1711, 1604, 1685, 1801, 1629, 1632, 1760, 1806, 1586, 1805, 1558, 1585, 1746, 1580, 1575, 1786, 1600, 1752, 1572, 1781, 1665, 1634, 1715, 1673, 1573, 1579, 1693, 1701, 1618, 1727]
DIAG_SEG_23_08_VAL = [1652, 1561, 1719, 1557, 1566, 1658, 1630, 1698, 1688, 1675, 1687, 1706, 1607, 1782, 1574, 1702, 1747, 1717, 1743, 1810, 1778, 1697, 1664, 1745, 1663, 1689, 1703, 1704, 1802, 1624, 1797, 1590, 1631, 1779, 1768, 1676, 1596, 1742, 1570, 1655, 1759, 1741, 1662, 1749, 1724, 1637, 1626, 1755, 1764, 1710, 1670, 1679]

TRAIN_FROM_23 = [id for id in range(1813, 2000)]
TEST_FROM_23 = [id for id in range(2000, 2084+1)]

# WHEN ADDING NEW PATIENTS, DON'T FORGET TO ADD THEM TO THE APROPRIATE DATASET CLASSES BELOW! (or improve this coding scheme)
#fmt: off

# The patients below are corrected in 2023_08_15, therefore they are ignored in 2023_07_09
CORRECTED = ['2023_07_09/Patient (1757)', '2023_07_09/Patient (1758)', '2023_07_09/Patient (1759)', '2023_07_09/Patient (1760)', '2023_07_09/Patient (1761)', '2023_07_09/Patient (1762)', '2023_07_09/Patient (1763)', '2023_07_09/Patient (1764)', '2023_07_09/Patient (1765)', '2023_07_09/Patient (1766)', '2023_07_09/Patient (1767)', '2023_07_09/Patient (1768)', '2023_07_09/Patient (1769)', '2023_07_09/Patient (1770)', '2023_07_09/Patient (1771)', '2023_07_09/Patient (1772)', '2023_07_09/Patient (1773)', '2023_07_09/Patient (1774)', '2023_07_09/Patient (1775)', '2023_07_09/Patient (1776)', '2023_07_09/Patient (1777)', '2023_07_09/Patient (1778)', '2023_07_09/Patient (1779)', '2023_07_09/Patient (1780)', '2023_07_09/Patient (1781)', '2023_07_09/Patient (1782)', '2023_07_09/Patient (1783)', '2023_07_09/Patient (1784)', '2023_07_09/Patient (1785)', '2023_07_09/Patient (1786)', '2023_07_09/Patient (1787)', '2023_07_09/Patient (1788)', '2023_07_09/Patient (1789)', '2023_07_09/Patient (1790)', '2023_07_09/Patient (1791)', '2023_07_09/Patient (1792)', '2023_07_09/Patient (1793)', '2023_07_09/Patient (1794)', '2023_07_09/Patient (1795)', '2023_07_09/Patient (1796)', '2023_07_09/Patient (1797)', '2023_07_09/Patient (1798)', '2023_07_09/Patient (1799)', '2023_07_09/Patient (1800)', '2023_07_09/Patient (1801)', '2023_07_09/Patient (1802)', '2023_07_09/Patient (1803)', '2023_07_09/Patient (1804)', '2023_07_09/Patient (1805)', '2023_07_09/Patient (1806)', '2023_07_09/Patient (1807)', '2023_07_09/Patient (1808)', '2023_07_09/Patient (1809)', '2023_07_09/Patient (1810)', '2023_07_09/Patient (1811)', '2023_07_09/Patient (1812)', '2023_07_09/Patient (1813)', '2023_07_09/Patient (1814)', '2023_07_09/Patient (1815)', '2023_07_09/Patient (1816)', '2023_07_09/Patient (1817)', '2023_07_09/Patient (1818)', '2023_07_09/Patient (1819)', '2023_07_09/Patient (1820)', '2023_07_09/Patient (1821)', '2023_07_09/Patient (1822)', '2023_07_09/Patient (1823)', '2023_07_09/Patient (1824)', '2023_07_09/Patient (1825)', '2023_07_09/Patient (1826)', '2023_07_09/Patient (1827)', '2023_07_09/Patient (1828)', '2023_07_09/Patient (1829)', '2023_07_09/Patient (1830)', '2023_07_09/Patient (1831)', '2023_07_09/Patient (1832)', '2023_07_09/Patient (1833)', '2023_07_09/Patient (1834)', '2023_07_09/Patient (1835)', '2023_07_09/Patient (1836)', '2023_07_09/Patient (1837)', '2023_07_09/Patient (1838)', '2023_07_09/Patient (1839)', '2023_07_09/Patient (1840)', '2023_07_09/Patient (1841)', '2023_07_09/Patient (1842)', '2023_07_09/Patient (1843)', '2023_07_09/Patient (1844)', '2023_07_09/Patient (1845)', '2023_07_09/Patient (1846)', '2023_07_09/Patient (1847)', '2023_07_09/Patient (1848)', '2023_07_09/Patient (1849)', '2023_07_09/Patient (1850)', '2023_07_09/Patient (1851)', '2023_07_09/Patient (1852)', '2023_07_09/Patient (1853)', '2023_07_09/Patient (1854)', '2023_07_09/Patient (1855)', '2023_07_09/Patient (1856)', '2023_07_09/Patient (1857)', '2023_07_09/Patient (1858)', '2023_07_09/Patient (1859)', '2023_07_09/Patient (1860)', '2023_07_09/Patient (1861)', '2023_07_09/Patient (1862)', '2023_07_09/Patient (1863)', '2023_07_09/Patient (1864)', '2023_07_09/Patient (1865)', '2023_07_09/Patient (1866)', '2023_07_09/Patient (1867)', '2023_07_09/Patient (1868)', '2023_07_09/Patient (1869)', '2023_07_09/Patient (1997)', '2023_07_09/Patient (1998)', '2023_07_09/Patient (1999)', '2023_07_09/Patient (2000)', '2023_07_09/Patient (2001)', '2023_07_09/Patient (2002)', '2023_07_09/Patient (2003)', '2023_07_09/Patient (2004)', '2023_07_09/Patient (2005)', '2023_07_09/Patient (2006)', '2023_07_09/Patient (2007)', '2023_07_09/Patient (2008)', '2023_07_09/Patient (2009)', '2023_07_09/Patient (2010)', '2023_07_09/Patient (2011)', '2023_07_09/Patient (2012)', '2023_07_09/Patient (2013)', '2023_07_09/Patient (2014)', '2023_07_09/Patient (2015)', '2023_07_09/Patient (2016)', '2023_07_09/Patient (2017)', '2023_07_09/Patient (2018)', '2023_07_09/Patient (2019)', '2023_07_09/Patient (2020)', '2023_07_09/Patient (2021)', '2023_07_09/Patient (2022)', '2023_07_09/Patient (2023)', '2023_07_09/Patient (2024)', '2023_07_09/Patient (2025)', '2023_07_09/Patient (2026)', '2023_07_09/Patient (2027)', '2023_07_09/Patient (2028)', '2023_07_09/Patient (2029)', '2023_07_09/Patient (2030)', '2023_07_09/Patient (2031)', '2023_07_09/Patient (2032)', '2023_07_09/Patient (2033)', '2023_07_09/Patient (2034)', '2023_07_09/Patient (2035)', '2023_07_09/Patient (2036)', '2023_07_09/Patient (2037)', '2023_07_09/Patient (2038)', '2023_07_09/Patient (2039)', '2023_07_09/Patient (2040)', '2023_07_09/Patient (2041)', '2023_07_09/Patient (2042)', '2023_07_09/Patient (2043)', '2023_07_09/Patient (2044)', '2023_07_09/Patient (2045)', '2023_07_09/Patient (2046)', '2023_07_09/Patient (2047)', '2023_07_09/Patient (2048)', '2023_07_09/Patient (2049)', '2023_07_09/Patient (2050)', '2023_07_09/Patient (2051)', '2023_07_09/Patient (2052)', '2023_07_09/Patient (2053)', '2023_07_09/Patient (2054)', '2023_07_09/Patient (2055)', '2023_07_09/Patient (2056)', '2023_07_09/Patient (2057)', '2023_07_09/Patient (2058)', '2023_07_09/Patient (2059)', '2023_07_09/Patient (2060)', '2023_07_09/Patient (2061)', '2023_07_09/Patient (2062)', '2023_07_09/Patient (2063)', '2023_07_09/Patient (2064)', '2023_07_09/Patient (2065)', '2023_07_09/Patient (2066)', '2023_07_09/Patient (2067)', '2023_07_09/Patient (2068)', '2023_07_09/Patient (2069)', '2023_07_09/Patient (2070)', '2023_07_09/Patient (2071)', '2023_07_09/Patient (2072)', '2023_07_09/Patient (2073)', '2023_07_09/Patient (2074)', '2023_07_09/Patient (2075)', '2023_07_09/Patient (2076)', '2023_07_09/Patient (2077)', '2023_07_09/Patient (2078)', '2023_07_09/Patient (2079)', '2023_07_09/Patient (2080)', '2023_07_09/Patient (2081)', '2023_07_09/Patient (2082)', '2023_07_09/Patient (2083)', '2023_07_09/Patient (2084)']

IGNORED_SAMPLES = [
    "2022_11_22/Patient (5)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (10)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (49)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (49)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (40)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (27)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (71)/T2_Mapping_/Apex.dcm",
    "2022_11_22/Patient (104)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (93)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (72)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (77)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (73)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (120)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (119)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (111)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (72)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (120)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (111)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (110)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (218)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (218)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (149)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (138)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (365)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (335)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (356)/T1_Mapping_/Apex.dcm",
    "2022_11_22/Patient (365)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (319)/T2_Mapping_/Base.dcm",
    "2022_11_22/Patient (306)/T2_Mapping_/Mid.dcm",
    "2022_11_22/Patient (444)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (416)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (390)/T1_Mapping_/Base.dcm",
    "2022_11_22/Patient (424)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (416)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (390)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (444)/T1_Mapping_/Mid.dcm",
    "2022_11_22/Patient (424)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1655)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1658)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1661)/T1_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1661)/T1_Mapping_/Mid.dcm",
    "2023_07_09/Patient (1661)/T2_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1663)/T2_Mapping_/Apex.dcm",
    "2023_07_09/Patient (1663)/T1_Mapping_/Mid.dcm",
] + CORRECTED


class MappingDatasetAlbu(Dataset):
    """ Pytorch dataset for the SE dataset that supports Albumentations augmentations (including bounding box safe cropping).

        Args:
            root (str, Path): Root directory of the dataset containing the 'Patient (...)' folders.
            transforms (albumentations.Compose, optional): Albumentations augmentation pipeline. Defaults to None.
            split (str, optional):  The dataset split, supports `train`, `val`, `test` or `interobserver`. Defaults to 'train'.
            check_dataset (bool, optional): Check dataset for missing and unexpected files. Outdated... Defaults to False.
            obser>ver_id (int, optional): Contours from 3 annotators are available for the same patient in the `interobserver` `split`. `observer_id` selects from these. Supports 1,2 or 3. Defaults to 1.
            mapping_only (bool, optional): Include T1 and T2 mappping images only or also include the map sequences that were used to construct these mappings. Defaults to False.
    """
    def __init__(self, 
                 root, 
                 transforms=None, 
                 split='train', 
                 check_dataset=False, 
                 observer_id = 1,
                 mapping_only = False) -> None:
        
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        if check_dataset:
            mapping_utils.check_datset(self.root)
        if split == "interobserver" and observer_id != 1:
            contours_filename = f"Contours_{observer_id}.json"
        else:
            contours_filename = f"Contours.json"
        self.all_samples = mapping_utils.construct_samples_list(
            self.root, contours_filename
        )
        mapping_utils.print_diagnostics(self.root, self.all_samples)

        segmentation_partitions = {
            "train": TRAIN_PATIENTS + DIAG_SEG_23_08_TRAIN + TRAIN_FROM_23 + TEST_FROM_23,
            "val": VAL_PATIENTS + DIAG_SEG_23_08_VAL + TEST_FROM_23,
            "test": TEST_PATIENTS,
            "interobserver": INTEROBSERVER_PATIENTS,
        }

        self.all_samples = self.remove_ignored(self.all_samples, IGNORED_SAMPLES, self.root)
    
        self.samples = mapping_utils.split_samples_list(
            self.all_samples, segmentation_partitions[self.split]
        )

        mapping_utils.print_diagnostics(self.root, self.samples, get_mean_std=True)

        if mapping_only:
            self.to_mapping_only()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        path, target_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None, use_modality_lut=False) # TODO: change to use_modality_lut=True
        target_contours = mapping_utils.load_contours(target_path)
        target = mapping_utils.contours_to_masks_v2(target_contours, sample.shape)
        if self.transforms is not None:
            if "bboxes" in self.transforms.processors.keys():
                bbox = self.compute_bbox(target)
                transformed = self.transforms(image=sample, mask=target, bboxes=[bbox])
            else:
                transformed = self.transforms(image=sample, mask=target)
            sample, target = transformed["image"], transformed["mask"]

        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        sample = np.expand_dims(sample, 0)

        return sample, target

    def compute_bbox(self, mask):
        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox = [
            mask.nonzero()[1].min(),  # / mask.shape[0],
            mask.nonzero()[0].min(),  # / mask.shape[1],
            mask.nonzero()[1].max(),  # / mask.shape[0],
            mask.nonzero()[0].max(),  # / mask.shape[1],
            "dummy_label",
        ]  # x_min, y_min, x_max, y_max
        return bbox

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]

    @staticmethod
    def remove_ignored(samples, ignore_list, dataset_root, verbose=False):
        samples_numpy = np.array(samples)
        remove_count = 0
        for sample in ignore_list:
            sample = os.path.join(dataset_root, sample)  
            # sample is like     "2022_11_22/Patient (5)/T1_Mapping_/Apex.dcm" 
            # map sample is like "2022_11_22/Patient (5)/T1_Mapping_/_map_apex/"
            # also sample can be a Patient folder, e.g.: "2022_11_22/Patient (5)"
            map_sample = sample.replace("_Mapping_/Apex.dcm", "_map_apex")
            map_sample = map_sample.replace("_Mapping_/Mid.dcm", "_map_mid_")
            map_sample = map_sample.replace("_Mapping_/Base.dcm", "_map_base")
            for s in [sample, map_sample]:
                samples_to_drop = np.char.startswith(samples_numpy[:,0], s)
                if samples_to_drop.any():
                    samples_numpy = samples_numpy[~samples_to_drop]
                    if verbose:
                        print(colored(f"Removed {s} from the dataset based on the IGNORED_SAMPLES list", "red"))
                    remove_count += np.sum(samples_to_drop)
        print(f"Removed {remove_count} samples from the dataset based on the IGNORED_SAMPLES list")
        return samples_numpy.tolist()


class MappingDiagnosisDatasetAlbu(Dataset):
    # The samples list is class attributes and initialized during the first intantiation.
    all_samples = None
    img_cache = None
    supported_classes = ["Negative", "HCM", "Athlete", "Amyloidosis", "Akut myocarditis", "Lezajlott myocarditis"]

    def __init__(
        self,
        root,
        transforms=None,
        split="train",
        check_dataset=False,
        stack_slices_by_type=False,
        num_classes=3,
        include_long_axis=True,
        diagnosis_csv="Mapping_data_diagnosis_raw.csv"
    ) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split
        self.stack_slices_by_type = stack_slices_by_type
        assert num_classes in range(2, len(self.supported_classes) + 1)
        self.num_classes = num_classes
        self.classes = self.supported_classes[:num_classes]
        self.diag_df = pd.read_csv(os.path.join(root, diagnosis_csv), index_col=0)
        if self.num_classes == 5:
            # Merge "Akut myocarditis" and "Lezajlott myocarditis" classes
            self.diag_df.loc[self.diag_df["Established diagnosis"].isin(["Akut myocarditis", "Lezajlott myocarditis"]), "Established diagnosis"] = "Myocarditis"
            self.classes = self.supported_classes[:4] + ["Myocarditis"]

        self.diag_df.loc[
            TRAIN_PATIENTS + HCM_TRAIN + ATHLETE_TRAIN + AMYLOIDOSIS_TRAIN + AKUT_MYOCARDITIS_TRAIN + LEZAJLOTT_MYOCARDITIS_TRAIN,
            "partition"
        ] = "train"
        self.diag_df.loc[VAL_PATIENTS + HCM_VAL + ATHLETE_VAL + AMYLOIDOSIS_VAL + AKUT_MYOCARDITIS_VAL + LEZAJLOTT_MYOCARDITIS_VAL, "partition"] = "val"
        # DANGEROUS MOVE: Add interobserver and test patients to the validation partition
        self.diag_df.loc[TEST_PATIENTS + HCM_TEST + ATHLETE_TEST + AKUT_MYOCARDITIS_TEST + LEZAJLOTT_MYOCARDITIS_TEST, "partition"] = "val"
        self.diag_df.loc[INTEROBSERVER_PATIENTS, "partition"] = "val"

        slices_names = [
            "T1_Mapping_/Apex.dcm",
            "T1_Mapping_/Base.dcm",
            "T1_Mapping_/Mid.dcm",
            "T2_Mapping_/Apex.dcm",
            "T2_Mapping_/Base.dcm",
            "T2_Mapping_/Mid.dcm",
            "T1_Mapping_long/2ch.dcm",
            "T1_Mapping_long/3ch.dcm",
            "T1_Mapping_long/4ch.dcm",
        ]
        if not include_long_axis:
           slices_names = slices_names[:-3]

        use_path_cache = True # DONT'T FORGET THE USE OF THE PATH CACHE!!! IT MIGH CAUSE TROUBLE!!! 
        diag_df_cache_file = f"diag_df_with_path_cache_{self.root.replace('/','_')}.csv"
        if os.path.exists(diag_df_cache_file) and use_path_cache:
            self.diag_df = pd.read_csv(diag_df_cache_file, index_col=0)  
            print("Diagnosis files path cache loaded from: ", diag_df_cache_file)  
        else:
            all_dcm_paths = mapping_utils.consturct_unlabelled_samples_list(self.root)
            for patient_id in tqdm(self.diag_df.index, desc="Finding files for patients"):
                patient_dcms = [dcm for dcm in all_dcm_paths if f"Patient ({patient_id})" in dcm]
                for slice in slices_names:
                    dcm_exists = False
                    for dcm_path in patient_dcms:
                        if slice in dcm_path:
                            self.diag_df.loc[patient_id, slice] = dcm_path
                            dcm_exists = True
                    if not dcm_exists:
                        self.diag_df.loc[patient_id, slice] = pd.NA
            if use_path_cache:
                self.diag_df.to_csv(diag_df_cache_file)

        print("Num. patients total:", len(self.diag_df))

        self.diag_df.to_csv("diag_df.csv")

        # Assigh an integer to each diagnosis class
        classes_dict = dict(zip(self.classes, range(len(self.classes))))
        self.diag_df["numeric_diagnosis"] = [
            classes_dict.get(diag) for diag in self.diag_df["Established diagnosis"]
        ]
        # Filter patients with valid, not None class, i.e. with classes of interest:
        self.diag_df.dropna(inplace=True, subset=["numeric_diagnosis"])
        print(f"Num. patients with classes {self.classes}:", len(self.diag_df))

        partition_stats = (
            self.diag_df.groupby("partition")["Established diagnosis"]
            .value_counts()
            .unstack()
        )
        #print(tabulate(partition_stats, headers=partition_stats.columns, tablefmt="pipe"))

        # There are no Positive validation patients in this dataset, reassign some of the train patients to the validation set
        # WARNING: This may distort results, as during pretraining the now val samples were train samples
        for i in [9, 26, 63, 71]:
            self.diag_df.loc[i, "partition"] = "val"

        # Drop patients where one of the mapping dicoms is missing
        self.diag_df.dropna(inplace=True, subset=slices_names)
        print("Num. patients with all mapping dicoms:", len(self.diag_df))

        for partition in ["train", "val", "test", "interobserver"]:
            print(
                f"Num. {partition} patients after filtering",
                sum(self.diag_df["partition"] == partition),
            )

        if self.img_cache is None:
            img_cache = {}
            for patient_id in tqdm(
                self.diag_df.index,
                total=len(self.diag_df),
                desc="Constucting samples cache from files",
            ):
                slices = []
                for slice in slices_names:
                    dcm_path = self.diag_df.loc[patient_id, slice]
                    slices.append(mapping_utils.load_dicom(dcm_path, mode=None))
                # slices = np.stack(slices, axis=0) # Channels first concatenation
                img_cache[patient_id] = slices
            MappingDiagnosisDatasetAlbu.img_cache = img_cache

        partition_stats = (
            self.diag_df.groupby("partition")["Established diagnosis"]
            .value_counts()
            .unstack()
        )
        #print(tabulate(partition_stats, headers=partition_stats.columns, tablefmt="pipe"))

        self.samples = self.diag_df[self.diag_df["partition"] == self.split]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        # dcm_path = self.samples[index][0]
        # sample = mapping_utils.load_dicom(dcm_path, mode=None)
        patient_id = self.samples.index[index]
        sample = self.img_cache[patient_id]

        if self.transforms is not None:
            transformed_sample = []
            for slice in sample:
                transformed = self.transforms(image=slice)
                transformed = transformed["image"]
                if not self.stack_slices_by_type:
                    # Convert images to channels_first mode, from albumentations' 2d grayscale images
                    transformed = np.expand_dims(transformed, 0)
                transformed_sample.append(transformed)
            sample = transformed_sample

        if self.stack_slices_by_type:
            # Stack T1 and T2 mapping records to be fed to the encoder together (T1 and T2 separately)
            sample = [
                np.stack(sample[0:3], axis=0),
                np.stack(sample[3:6], axis=0),
                np.stack(sample[6:9], axis=0),
            ]

        # Get target
        # e.g. path_parts: .../Patient (1)/T1_map_base/Contours.json -> Patient (1)
        target = self.diag_df.loc[patient_id, "numeric_diagnosis"]
        # if self.num_classes > 2:
        #     target = utils.to_categorical_np(target, self.num_classes)

        # Target should be a vector, even if it is one dimensional
        target = np.array(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class MappingUnlabeledDataset(Dataset):

    def __init__(self, 
                 root, 
                 transforms=None, 
                 split='train', 
                 mapping_only = False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.split = split

        self.all_samples = mapping_utils.consturct_unlabelled_samples_list(self.root)
        mapping_utils.print_diagnostics(self.root, self.all_samples)
        
        
        # UNLABELED = list(range(481, 1262)) # patients without any label (neither segmentation nor diagnosis)
        unlabeled_partitions = {
            "train": TRAIN_PATIENTS + HCM_TRAIN + ATHLETE_TRAIN + AMYLOIDOSIS_TRAIN + AKUT_MYOCARDITIS_TRAIN + LEZAJLOTT_MYOCARDITIS_TRAIN + UNLABELED,
            "val": VAL_PATIENTS + HCM_VAL + ATHLETE_VAL + AMYLOIDOSIS_VAL + AKUT_MYOCARDITIS_VAL + LEZAJLOTT_MYOCARDITIS_VAL,
            "test": TEST_PATIENTS + HCM_TEST + ATHLETE_TEST + AKUT_MYOCARDITIS_TEST + LEZAJLOTT_MYOCARDITIS_TEST,
        }

        self.samples = mapping_utils.split_samples_list(
            self.all_samples, unlabeled_partitions[self.split]
        )
        if mapping_only:
            self.to_mapping_only()
        print("Unlabeled dataset size:", len(self), "images")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target)
        """
        path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None)
        if self.transforms is not None:
            if isinstance(self.transforms, FullTransformPipeline):
                # Division by 16: 4096/255 = 16 - scaling 14 bit images to 8 bits
                sample = Image.fromarray((sample//16).astype(np.uint8))
            sample = self.transforms(sample)
    
        return sample, 0

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [x for x in self.samples if "_Mapping_" in x]

if __name__ == "__main__":
    root = "/data/se"
    # root = "/home1/ssl-phd/data/mapping"
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    #import albumentations as A
    #from albumentations.pytorch import ToTensorV2

    # diag = MappingDiagnosisDatasetAlbu("cropped_mapping")
    # exit() 

    '''
    transform = A.Compose(
        [
            # A.LongestMaxSize(192),
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.Normalize(mean=(243.0248,), std=(391.5486,)),
            # ToTensorV2()
        ]
    )
    '''
    train_ds = MappingDatasetAlbu(
        root,
        None,
        check_dataset=False,
        mapping_only=False,
    )
    print("Train samples =", len(train_ds))
    val_ds = MappingDatasetAlbu(
        root, split="val", mapping_only=False
    )
    print("Val samples =", len(val_ds))
    test_ds = MappingDatasetAlbu(
        root, split="test", mapping_only=False
    )
    print("Test samples =", len(test_ds))
    interobs_ds = MappingDatasetAlbu(
        root, split="interobserver", mapping_only=False
    )

    # from torch.utils.data import DataLoader

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # print("Train dataloader = ", len(train_loader))

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=32,
    #     num_workers=2,
    #     shuffle=False,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    # print("Val dataloader = ", len(val_loader))

    # train_ds.to_mapping_only()
    print("Train samples =", len(train_ds))
    print("Val samples =", len(val_ds))
    print("Test samples =", len(test_ds))
    print("Interobserver samples =", len(interobs_ds))
    # print("Train dataloader = ", len(train_loader))
    # print("Val dataloader = ", len(val_loader))

    # img, mask = train_ds[10]
    # print("Img shape", img.shape)
    # print("Mask shape", mask.shape)
    # img = img[0,...]#.detach().numpy()
    # cv2.imwrite(f"example_transformed.png", np.concatenate([img, np.ones((img.shape[0], 3))*255, mask], axis=1))
