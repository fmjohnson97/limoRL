import cv2
import json

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

    breakpoint()
    labels['train']=train
    labels['test']=test
    labels['val']=val

    with open(label_path, 'w') as f:
        json.dump(labels,f)



if __name__=='__main__':
    # code to go from videos to data usable for nodeDataset.py

    # put the video frames into their node folders
    # video_files = glob('nodeVideos/*')
    # for vfile in tqdm(video_files):
    #     save_path = vfile.split('/')[-1].split('.')[0]
    #     saveVideoFrames(vfile,'nodePhotos/'+save_path+'/')

    # interpolate angle labels for images on the nodes
    # photo_folders = glob('nodePhotos/*')
    # for folder in photo_folders:
    #     interpolateFrameAngleLabels(folder+'/labels.json')
    #     createLabelSplits(folder+'/labels.json')

    interpolateFrameAngleLabels('nodePhotos/node9/labels.json')
    createLabelSplits('nodePhotos/node9/labels.json')

    #TODO: resize all images