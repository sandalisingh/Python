# python3 Covid/Covid.py

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_excel('./Covid/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
print(data.head(50))

