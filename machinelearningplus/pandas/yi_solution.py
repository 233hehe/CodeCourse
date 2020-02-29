"""
problem set from https://www.machinelearningplus.com/python/101-pandas-exercises-python/
"""
from pprint import pformat
from typing import List, Dict, Any

# 1. How to import pandas and check the version?
import pandas as pd
#print(pd.__version__)

# 2. How to create a series from a list, numpy array and dict?
import numpy as np
mylist_2: List[str] = list('abcedfghijklmnopqrstuvwxyz')
myarr_2: np.ndarray = np.arange(26)
mydict_2: Dict[str, int] = dict(zip(mylist_2, myarr_2))

mylist_series_2: pd.Series = pd.Series(mylist_2)
myarr_series_2: pd.Series = pd.Series(myarr_2)
mydict_series_2: pd.Series = pd.Series(mydict_2)
#print(mylist_series_2.head(), myarr_series_2.head(), mydict_series_2.head())

# 3. How to convert the index of a series into a column of a dataframe?
mylist_3: List[str] = list('abcedfghijklmnopqrstuvwxyz')
myarr_3: np.ndarray = np.arange(26)
mydict_3: Dict[str, int] = dict(zip(mylist_3, myarr_3))
ser_3: pd.Series = pd.Series(mydict_3)

pdf_3: pd.DataFrame = ser_3.to_frame()
#print(pdf_3.head(5))

# 4. How to combine many series to form a dataframe?
ser1_4: pd.Series = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2_4: pd.Series = pd.Series(np.arange(26))

pdf_4: pd.DataFrame = pd.concat([ser1_4, ser2_4])
#print(pdf_4.head(5))

# 5. How to assign name to the series’ index?
# Give a name to the series ser calling it ‘alphabets’.
ser_5: pd.Series = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser_5.name = "alphabets"
#print(ser_5)

# 6. How to get the items of series A not present in series B?
# From ser1 remove items present in ser2.
ser1_6: pd.Series = pd.Series([1, 2, 3, 4, 5])
ser2_6: pd.Series = pd.Series([4, 5, 6, 7, 8])
ser_6: pd.Series = ser1_6[~ser1_6.isin(ser2_6)]
# print(ser_6)


# 7. How to get the items not common to both series A and series B?
# Get all items of ser1 and ser2 not common to both.
ser1_7 = pd.Series([1, 2, 3, 4, 5])
ser2_7 = pd.Series([4, 5, 6, 7, 8])
ser_7 = ser1_7[ser1_7.isin(ser2_7)]
# print(ser_7)



# 8. How to get the minimum, 25th percentile, median, 75th, and max of a numeric series?
# Compute the minimum, 25th percentile, median, 75th, and maximum of ser.
ser_8 = pd.Series(np.random.normal(10, 5, 25))
# print(np.percentile(ser_8, [0, 0.25, 0.5, 0.75, 1]))

#9. How to get frequency counts of unique items of a series?
# Calculte the frequency counts of each unique value ser.
ser_9 = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
ser_out_9 = np.unique(ser_9, return_counts=True)
# print(ser_out_9)
# print(ser_9.value_counts())

