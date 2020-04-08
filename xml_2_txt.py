import xml.etree.cElementTree as et   # 读取xml文件的包
import os

def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


data_dir1='./data1209'
data_dir2='./data1203'
ratio=0.9

xml_list1=[]
GetFileList(data_dir1,xml_list1)
xml_list1=[x for x in xml_list1 if 'xml' in x]

xml_list2=[]
GetFileList(data_dir2,xml_list2)
xml_list2=[x for x in xml_list2 if 'xml' in x]

xml_list=xml_list1+xml_list2


xml_list=list(set(xml_list))
train_list=xml_list[:int(len(xml_list)*ratio)]
val_list=xml_list[int(len(xml_list)*ratio):]

train_file=open('train.txt',mode='w')
val_file=open('val.txt',mode='w')


for xml_name in train_list:
    try:
        tree = et.parse(xml_name)
    except:
        print(xml_name,'err')
        continue
    root = tree.getroot()  # 使用getroot()获取根节点，得到的是一个Element对象

    img_name=root.find('filename').text

    print(img_name)
    tmp_str=''
    img_path=xml_name.replace('.xml','.jpg')
    tmp_str+=img_path+'|'


    obj=root.find('object')


    label=obj.find('name').text

    if label=='qrcode':

        xml_box = obj.find('bndbox')
        xmin = (int(float(xml_box.find('xmin').text)) )
        ymin = (int(float(xml_box.find('ymin').text)) )
        xmax = (int(float(xml_box.find('xmax').text)) )
        ymax = (int(float(xml_box.find('ymax').text)) )

        tmp_str+=' %d,%d,%d,%d,%d'%(xmin,ymin,xmax,ymax,1)

    tmp_str+='\n'

    train_file.write(tmp_str)

train_file.close()

for xml_name in val_list:
    try:
        tree = et.parse(xml_name)
    except:
        continue
    root = tree.getroot()  # 使用getroot()获取根节点，得到的是一个Element对象

    img_name = root.find('filename').text

    tmp_str = ''
    img_path=xml_name.replace('.xml','.jpg')
    tmp_str += img_path + '|'

    obj = root.find('object')
    label = obj.find('name').text

    if label == 'qrcode':
        xml_box = obj.find('bndbox')
        xmin = (int(float(xml_box.find('xmin').text)))
        ymin = (int(float(xml_box.find('ymin').text)))
        xmax = (int(float(xml_box.find('xmax').text)))
        ymax = (int(float(xml_box.find('ymax').text)) )



        tmp_str += ' %d,%d,%d,%d,%d' % (xmin, ymin, xmax, ymax, 1)

    tmp_str += '\n'

    val_file.write(tmp_str)

val_file.close()
