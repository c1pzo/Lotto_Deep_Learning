import pandas as pd

read_file = pd.read_csv (r'/home/c1pzo/test/GV_MASTER.txt')
read_file.to_csv (r'/home/c1pzo/test/GV_MASTER.csv', index=None)
