from pathlib import Path
import os
import pandas_datareader as pdr
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
path = os.path.join(BASE_DIR, 'Files', 'Temp.xlsx')
#print(str(path))
nifty_index = pdr.get_data_yahoo('^NSEI', datetime(2007, 1, 1), datetime(2021, 5, 1), interval='m')
print(nifty_index)