import os
import cv2
import json

from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

def saveVideoFrames(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    img_num = 0
    base_name = save_path.split('/')[-2]
    # breakpoint()
    while ret == True:
        frame = cv2.resize(frame,(frame.shape[1]//4, frame.shape[0]//4))
        cv2.imwrite(save_path+base_name+'_im'+str(img_num)+'.png', frame)
        img_num+=1
        ret, frame = cap.read()
    cap.release()

def interpolateFrameAngleLabels(label_path):
    with open(label_path) as f:
        labels = json.load(f)

    im_names = list(labels.keys())
    im_num = int(im_names[0].split('m')[-1])
    temp = labels[im_names[0]]
    x, y, z = temp[0], temp[1], temp[2]

    new_labels = {}
    for i in range(len(im_names)-1):
        im2_num = int(im_names[i+1].split('m')[-1])
        temp2 = labels[im_names[i+1]]
        im_diff = im2_num - im_num
        angle_diff = temp2[-1] - temp[-1]

        delta = angle_diff/im_diff
        angle = temp[-1]
        while im_num<im2_num:
            new_labels["im" + str(im_num)] = [x, y, z, int(angle)]
            angle+=delta
            im_num+=1

        temp = temp2
        im_num = im2_num
    new_labels["im" + str(im_num)] = [x, y, z, int(angle)]


    # breakpoint()
    with open(label_path, "w") as f:
        json.dump(new_labels, f)

def countPhotosPerAngle(label_path):
    with open(label_path) as f:
        labels = json.load(f)

    angles=defaultdict(list)
    # ang_list = []
    for key, val in labels.items():
        angles[val[-1]].append(key)
        # ang_list.append(val[-1])
    # plt.hist(ang_list, bins= range(361))
    # plt.show()

    return angles

def createLabelSplits(label_path):
    angles = countPhotosPerAngle(label_path)
    train = []
    test = []
    val = []
    addTest = False
    addVal = True
    keys = list(angles.keys())
    key_ind = 0

    while len(angles)>0:
        key = keys[key_ind]
        train.append(angles[key][0])
        if key_ind % 5 == 0:
            addTest = True
        if key_ind % 10 == 0:
            addVal = True
        if len(angles[key])<2:
            # breakpoint()
            angles.pop(key)
        else:
            angles[key]=angles[key][1:]
            if addTest:
                test.append(angles[key][0])
                addTest = False
                angles[key] = angles[key][1:]
            if addVal and len(angles[key])>0:
                val.append(angles[key][0])
                addVal = False
                angles[key] = angles[key][1:]
            if len(angles[key])<1:
                # breakpoint()
                angles.pop(key)
        key_ind+=1
        if key_ind >= len(keys):
            # breakpoint()
            key_ind=0
            keys=list(angles.keys())

    with open(label_path) as f:
        labels = json.load(f)

    train_angs = []
    for im in train:
        train_angs.append(labels[im][-1])

    test_angs = []
    for im in test:
        test_angs.append(labels[im][-1])

    val_angs = []
    for im in val:
        val_angs.append(labels[im][-1])

    print(label_path)
    plt.figure()
    plt.hist(train_angs, bins=range(361))
    plt.title('train angles')

    plt.figure()
    plt.hist(test_angs, bins=range(361))
    plt.title('test angles')

    plt.figure()
    plt.hist(val_angs, bins=range(361))
    plt.title('val angles')
    plt.show()

    # breakpoint()
    labels['train']=train
    labels['test']=test
    labels['val']=val

    with open(label_path, 'w') as f:
        json.dump(labels,f)

def makeNodeAngleListFile(node_photo_folder):
    node_angle_key={}
    for node_folder in node_photo_folder:
        with open(node_folder+'/labels.json', 'r') as f:
            labels = json.load(f)
        labels.pop('train')
        labels.pop('val')
        labels.pop('test')
        breakpoint()
        angle_list = defaultdict(list)
        for k,v in labels.items():
            angle_list[v[-1]].append(k)

        node_angle_key[node_folder.split('node')[-1]]=angle_list

    with open('labGraphAngleKey.json', 'w') as f:
        json.dump(node_angle_key,f)

def resizeAllImages(node_photo_folder, new_folder):
    all_photo_files = glob(node_photo_folder+'*/*.png')
    for file in all_photo_files:
        with Image.open(file) as img:
            img_new =img.resize((img.width // 4, img.height // 4))
            if not os.path.exists(new_folder+file.split('/')[1]+'/'):
                os.makedirs(new_folder+file.split('/')[1]+'/')
            img_new.save(new_folder+file.split('Photos')[-1][1:])

def copyLabelsJsonFiles(old_node_folder, new_node_folder):
    all_node_folders = glob(old_node_folder+'*')
    for folder in all_node_folders:
        with open(folder+'/labels.json') as f:
            labels = json.load(f)
        with open(new_node_folder+folder.split('/')[1]+'/labels.json','w') as f:
            json.dump(labels, f)

if __name__=='__main__':
    # code to go from videos to data usable for nodeDataset.py

    # put the video frames into their node folders
    # video_files = glob('nodeVideos/*')
    # for vfile in tqdm(video_files):
    #     save_path = vfile.split('/')[-1].split('.')[0]
    #     saveVideoFrames(vfile,'nodePhotosSmall/'+save_path+'/')

    # interpolate angle labels for images on the nodes
    photo_folders = glob('nodePhotosSmall/*')
    # for folder in tqdm(photo_folders):
    #     try:
    #         interpolateFrameAngleLabels(folder+'/labels.json')
    #         createLabelSplits(folder+'/labels.json')
    #     except:
    #         pass
    #
    makeNodeAngleListFile(photo_folders)

    # resize all the images and copy over the labels json files
    # resizeAllImages('nodePhotos/', 'nodePhotosSmall/')
    # copyLabelsJsonFiles('nodePhotosSmall_old/', 'nodePhotosSmall/')