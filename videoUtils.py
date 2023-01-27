import cv2

from glob import glob
from tqdm import tqdm

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






if __name__=='__main__':
    video_files = glob('nodeVideos/*')
    for vfile in tqdm(video_files):
        save_path = vfile.split('/')[-1].split('.')[0]
        saveVideoFrames(vfile,'nodePhotos/'+save_path+'/')