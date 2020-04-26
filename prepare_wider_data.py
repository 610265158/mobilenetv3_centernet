#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os



WIDER_ROOT = './WIDER'
train_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                               'wider_face_train_bbx_gt.txt')
val_list_file = os.path.join(WIDER_ROOT, 'wider_face_split',
                             'wider_face_val_bbx_gt.txt')

WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')


def parse_wider_file(root, file):
    with open(file, 'r') as fr:
        lines = fr.readlines()
    face_count = []
    img_paths = []
    face_loc = []
    img_faces = []
    count = 0
    flag = False
    for k, line in enumerate(lines):
        line = line.strip().strip('\n')
        if count > 0:
            line = line.split(' ')
            count -= 1
            loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
            face_loc += [loc]
        if flag:
            face_count += [int(line)]
            flag = False
            count = int(line)
        if 'jpg' in line:
            img_paths += [os.path.join(root, line)]
            flag = True

    total_face = 0
    for k in face_count:
        face_ = []
        for x in range(total_face, total_face + k):
            face_.append(face_loc[x])
        img_faces += [face_]
        total_face += k
    return img_paths, img_faces


def wider_data_file():
    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open('train.txt', 'w')
    for index in range(len(img_paths)):
        tmp_str = ''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,0'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data
        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open('val.txt', 'w')
    for index in range(len(img_paths)):

        tmp_str=''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,0'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data



        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()






def wider_data_file_refine():

    def parse_wider_file(root, file):
        with open(file, 'r') as fr:
            lines = fr.readlines()
        face_count = []
        img_paths = []
        face_loc = []
        img_faces = []
        count = 0


        one_image_faces=[]
        for k, line in enumerate(lines):
            if "#" in line:

                img_paths += [os.path.join(root, line[2:].rstrip())]

                one_image_faces=[]
            if '#' not in line:
                line = line.strip().strip('\n')


                line = line.split(' ')

                loc = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]

                one_image_faces.append(loc)


            if k<len(lines)-1:
                if '#' in lines[k+1] :
                    img_faces.append(one_image_faces.copy())
            else:
                img_faces.append(one_image_faces.copy())

        return img_paths, img_faces





    WIDER_ROOT = './WIDER'
    train_list_file = os.path.join(WIDER_ROOT, 'refine', 'train',
                                   'label.txt')
    val_list_file = os.path.join(WIDER_ROOT, 'refine', 'val',
                                 'label.txt')

    WIDER_TRAIN = os.path.join(WIDER_ROOT, 'WIDER_train', 'images')
    WIDER_VAL = os.path.join(WIDER_ROOT, 'WIDER_val', 'images')






    img_paths, bbox = parse_wider_file(WIDER_TRAIN, train_list_file)
    fw = open('train.txt', 'w')
    for index in range(len(img_paths)):
        tmp_str = ''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,0'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data
        if len(boxes) == 0:
            print(tmp_str)
            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()

    img_paths, bbox = parse_wider_file(WIDER_VAL, val_list_file)
    fw = open('val.txt', 'w')
    for index in range(len(img_paths)):

        tmp_str=''
        tmp_str =tmp_str+ img_paths[index]+'|'
        boxes = bbox[index]

        for box in boxes:
            data = ' %d,%d,%d,%d,0'%(box[0], box[1], box[0]+box[2],  box[1]+box[3])
            tmp_str=tmp_str+data



        if len(boxes) == 0:

            continue
        ####err box?
        if box[2] <= 0 or box[3] <= 0:
            pass
        else:
            fw.write(tmp_str + '\n')
    fw.close()



if __name__ == '__main__':

    try:
        wider_data_file_refine()

    except:
        wider_data_file()