import glob
import torch
import json
from torch.utils.data import random_split


def main():
    path = r'E:\code\Python\repo1\HRdepth-trans\dataset'
    files = glob.glob(path + '/Images/samples/*.json')
    shape_dict = {'shapes': None}
    for file in list(files):
        with open(file, 'r', newline='\n', encoding='utf-8') as fp:
            json_data = json.load(fp)
            shapes = json_data['shapes']
        for shape in shapes:
            x_min, y_min = shape['points'][0]
            x_max, y_max = shape['points'][1]
            if x_min > x_max:
                shape['points'][0][0] = x_max
                shape['points'][1][0] = x_min
            if y_min > y_max:
                shape['points'][0][1] = y_max
                shape['points'][1][1] = y_min

        shape_dict['shapes'] = shapes
        with open(path + '/Annotations/samples/' +file.split('\\')[-1], 'w', newline='\n', encoding='utf-8') as fp:
            json.dump(shape_dict, fp, ensure_ascii=False, indent=2)
    

if __name__ == "__main__":
    main()