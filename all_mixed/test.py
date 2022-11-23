import sys

sys.path.append('./all_mixed')
from datanetAPI import DatanetAPI

api = DatanetAPI('./data/all_mixed/train/')
it = iter(api)

num_sample = 0
for sample in it:
    print(num_sample)
    T = sample.get_traffic_matrix()
    print(T)
    num_sample+=1
print(num_sample)