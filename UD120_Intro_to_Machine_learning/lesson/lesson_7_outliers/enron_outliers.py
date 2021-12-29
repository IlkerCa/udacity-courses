#!/usr/bin/python3

import joblib
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np


### read in data dictionary, convert to numpy array

original = "../final_project/final_project_dataset.pkl"
destination = "../final_project/final_project_dataset.pkl"

'''
def unix_version(path_origin, path_dest):
    content = ''
    outsize = 0
    with open(path_origin, 'rb') as infile:
        content = infile.read()
        with open(path_dest, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
    return path_dest


pickled_file = unix_version(original, destination)
'''

data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop( 'TOTAL', 0 ) 
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
for key in data_dict:

    if data_dict[key]['exercised_stock_options'] == 'NaN':
        data_dict[key]['exercised_stock_options'] = np.nan
        data_dict[key]['salary'] = np.nan
        #print(key)
        


for key in data_dict:
    if (data_dict[key]['exercised_stock_options'] is not np.nan and data_dict[key]['salary'] is not np.nan) and (int(data_dict[key]['exercised_stock_options']) > 5000000 ) and int(data_dict[key]['salary']) > 1000000:
        print(key)

### your code below



