import os
import glob


def main():
    path = r'E:\code\Python\repo1\HRdepth-trans\dataset\kitti'
    os.chdir(path)
    
    with open('./split_.txt', 'r') as f:
        lines = f.readlines()
    # for line in lines:
        # print(line)
    content = ''
    for line in lines:
        content = content + line.split(' ')[0] + '\n'
    with open('./split_.txt', 'w')as f:
        f.write(content)

if __name__ == '__main__':
    main()