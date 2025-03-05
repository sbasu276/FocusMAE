import os
import shutil
from os.path import join

import torch
import PIL
import numpy as np

import cv2
import matplotlib.pyplot as plt
import imageio
# from data.data_transforms import AddInverse
from interpretability.utils import grad_to_img
from PIL import Image
# from project_utils import Str2List, to_numpy

def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()

def load_video(path, relative_box=[0., 1., 0., 1.], relative_times=[0., 1.]):

    # Opens the Video file
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    frames = []
    w1, w2, h1, h2 = relative_box
    start, end = relative_times
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        frames.append(frame[int(h1*h):int(h2*h), int(w1*w):int(w2*w)])

    cap.release()
    cv2.destroyAllWindows()
    total_frames = len(frames)

    return frames[int(start*total_frames):int(end*total_frames)], fps


@torch.no_grad()
def most_predicted(model, video, img_transforms):
    predictions = []
    for img in video:
        img = img_transforms(PIL.Image.fromarray(img)).cuda()[None]
        predictions.append((model(AddInverse()(img)))[0].argmax().item())
    c_idcs, counts = np.unique(predictions, return_counts=True)
    c_idx = c_idcs[np.argsort(counts)[-1]]

    print("Most predicted class:", c_idx)
    return c_idx


def process_video(model, img_transforms, video, class_idx=-1):
    if class_idx == -1:
        print("No class index provided, calculating most predicted class.")
        class_idx = most_predicted(model, video, img_transforms=img_transforms)

    atts = []
    imgs = []
    for img in video:
        model.zero_grad()
        img = AddInverse()(img_transforms(PIL.Image.fromarray(img)).cuda()[:][None]).requires_grad_(True)
        out = model(img)[0, class_idx]
        out.backward()
        att = grad_to_img(img[0], img.grad[0], alpha_percentile=100, smooth=5)
        att[..., -1] *= to_numpy(out.sigmoid())
        atts.append(to_numpy(att))
        imgs.append(np.array(to_numpy(img[0, :3].permute(1, 2, 0)) * 255, dtype=np.uint8))

    return imgs, atts


def save_video(imgs, atts, fps, id,gif_name="my.gif", path="gifs", dpi=75, imsize=224):
    folder = "tmp"
    os.makedirs(join(path, folder), exist_ok=True)
    for idx in range(atts.shape[0]):
        fig, ax = plt.subplots(1, figsize=(8, 4))
        plt.imshow(np.uint8(imgs[idx,...]*255/np.max(imgs[idx,...])), extent=(0, imsize, 0, imsize), )  ##
        plt.imshow(np.uint8(atts[idx, ...]*255), extent=(imsize, 2 * imsize, 0, imsize))

        # img1 = Image.fromarray(np.uint8(imgs[idx,...]*255/ np.max(imgs[idx,...])))

        # img2 = Image.fromarray(np.uint8(atts[idx, ...]) ,'RGBA' )

        # img1 = img1.resize((400, 400))

        # img2 = img2.resize((400,400)) dd
        # img1_size = img1.size
        # img2_size = img2.size 
        # dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        # dst.paste(img1, (0, 0))
        # dst.paste(img2, (img1.width, 0))
        plt.xlim(0, 2 * imsize)
        plt.xticks([])
        plt.yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if not os.path.exists(join(path, folder)):
            os.makedirs(join(path, folder))
        try:
            # dst.save(join(path, folder, "idx_{:03d}.png".format(idx)))
            plt.savefig(join(path, folder, "idx_{:03d}.png".format(idx)), bbox_inches='tight', dpi=dpi)
            
        except:
            print("frame not saved", idx)
            pass
        plt.close()
        
    images = []
    kargs = { 'fps': 5, 'quality': 10, 'macro_block_size': None, 
    'ffmpeg_params': ['-s','900x450'] }
    # filenames = [join(path, folder, "idx_{:03d}.png".format(idx)) for idx in range(len(atts))]
    filenames = sorted(os.listdir(join(path, folder)))
    for filename in filenames:
        if os.path.isfile(join(path, folder, filename)):
            try:
                images.append(imageio.imread(join(path, folder, filename)))
                os.remove(join(path, folder, filename))
            except:
                print("error in filenemame skipping")
                # os.remove(join(path, folder, filename))
    imageio.mimsave(join(path, gif_name), images,'MP4', **kargs)

    print(f"GIF saved under {join(path, gif_name)}")
    
    
    shutil.rmtree(join(path, folder), ignore_errors=True)


def save_video_WRONG(imgs, atts, fps, id,gif_name="my.gif", path="gifs", dpi=100, imsize=224):
    folder = "tmp"

    os.makedirs(join(path, folder), exist_ok=True)

    vid_writer = cv2.VideoWriter(os.path.join(path, gif_name), cv2.VideoWriter_fourcc(*'DIVX'), 5, (224,224))
    
    for idx in range(atts.shape[0]):


        frame = imgs[idx,...]

        frame = np.uint8((frame - np.min(frame))*255 / (np.max(frame)- np.min(frame)))
        att = cv2.cvtColor(atts[idx,...],cv2.COLOR_RGBA2RGB)
        # att = cv2.cvtcolor(atts[idx, ...],cv2.COLOR_BGRA2RGBA)
        print(att)
        att = (att - np.min(att))*255 / (np.max(att) - np.min(att))
        saliency_colormap = cv2.applyColorMap(np.uint8(att), cv2.COLORMAP_JET)
        smap = cv2.addWeighted(frame, 0.65, saliency_colormap, 0.35, 0)
        smap = np.uint8((smap - np.min(smap))*255./( np.max(smap)- np.min(smap)))
        vid_writer.write(smap)

        print(f"GIF saved under {join(path, gif_name)}")

    vid_writer.release()


    
    
    
    # shutil.rmtree(join(path, folder), ignore_errors=True)
