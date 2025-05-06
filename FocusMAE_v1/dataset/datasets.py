# pylint: disable=line-too-long,too-many-lines,missing-docstring
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from . import video_transforms, volume_transforms
from .loader import get_image_loader, get_video_loader
from .random_erasing import RandomErasing

import cv2

def annotation_convolve(string_anno,num_frames , stride=4 , seg_len=16):
    if not string_anno:
        return [0]*num_frames

    funcky = lambda x : int(x)
    anno = string_anno.strip().split(" ")
    anno = list(map(funcky, anno))

    return anno


def len_vid(sample):
    data = cv2.VideoCapture(sample) 
  
    # count the number of frames 
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT) 
    # fps = data.get(cv2.CAP_PROP_FPS) 
    return int(frames)

class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root='',
                 mode='train',
                 clip_len=8,
                 frame_sample_rate=2,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment=1,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 sparse_sample=False,
                 args=None):
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.sparse_sample = sparse_sample
        self.args = args
        self.aug = False
        self.rand_erase = False

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.video_loader = get_video_loader()

        cleaned = pd.read_csv(self.anno_path,header=None, delimiter=' ')
        
        self.dataset_samples = list(
            cleaned[0].apply(lambda row: os.path.join(self.data_root, row)))

        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)

                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

            

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            # T H W C
            buffer = self.load_video(sample, sample_rate_scale=scale_t)
            
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.load_video(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_video(sample)
            buffer = self.data_transform(buffer)

          
            return buffer, self.label_array[index], sample.split(
                "/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_video(sample)

            while len(buffer) == 0:
                warnings.warn(
                    "video {}, temporal {}, spatial {} not found during testing"
                    .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_video(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.sparse_sample:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_start = chunk_nb
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::self.test_num_segment, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]
            else:
                
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_step = max(
                    1.0 * (buffer.shape[0] - self.clip_len) /
                    (self.test_num_segment - 1), 0)
                temporal_start = int(chunk_nb * temporal_step)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split(
                "/")[-1].split(".")[0], chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            # crop_size=224,
            crop_size=args.input_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample

        try:
            vr = self.video_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)

        if self.mode == 'test' :
            if self.sparse_sample:
                tick = length / float(self.num_segment)
                all_index = []
                for t_seg in range(self.test_num_segment):
                    tmp_index = [
                        int(t_seg * tick / self.test_num_segment + tick * x)
                        for x in range(self.num_segment)
                    ]
                    all_index.extend(tmp_index)
                all_index = list(np.sort(np.array(all_index)))
            else:
                all_index = [
                    x for x in range(0, length, self.frame_sample_rate)
                ]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])

            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation':
                    end_idx = (converted_len + seg_len) // 2
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)



class VideoClsDataset_v2(Dataset):
    """Load your own video classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root='',
                 mode='train',
                 clip_len= 4,
                 frame_sample_rate=2,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment = 3,
                 step_size = 8,
                 num_crop=1,
                 test_num_segment=10,
                 test_num_crop=3,
                 sparse_sample=False,
                 transform = None,
                 args=None):
        
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.sparse_sample = sparse_sample
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.step_size = step_size

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.video_loader = get_video_loader()
        

        cleaned = pd.read_csv(self.anno_path,header=None, delimiter=' ')

        
        self.dataset_samples = list(
            cleaned[0].apply(lambda row: os.path.join(self.data_root, row)))
        
        self.label_array = list(cleaned.values[:, 1])
    
        
        if args.masking:
            self.mask_generator  = transform
        else:
            self.mask_generator = None

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


            self.val_num_segment  = args.num_sample
            self.val_num_crop = 1
        
            self.val_seg = []
            self.val_dataset = []
            self.val_label_array = []
            for ck in range(self.val_num_segment):
                for cp in range(self.val_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        if isinstance(sample_label,list):
                            val_seg_len = len(sample_label)//self.val_num_segment
                            if val_seg_len<1:
                                pass
                            self.val_label_array.append(np.max(sample_label[ck*val_seg_len:(ck+1)*val_seg_len ]))
                        else:
                            self.val_label_array.append(sample_label)
                        
                        self.val_dataset.append(self.dataset_samples[idx])
                        self.val_seg.append((ck, cp))
            


        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        

            ])
            
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            self.test_indices = {}
            if args.test_randomization: 

                for ck in range(self.test_num_segment):
                    for cp in range(self.test_num_crop):
                        for idx in range(len(self.label_array)):
                            sample_label = self.label_array[idx]
                            sample = self.dataset_samples[idx]
                            self.test_indices[sample] = []

                            if isinstance(sample_label, list):
                                test_seg_len = len(sample_label)//self.test_num_segment
                                if test_seg_len<1:
                                    pass
                                self.test_label_array.append(np.max(sample_label[ck*test_seg_len:(ck+1)*test_seg_len ]))
                            else:
                                self.test_label_array.append(sample_label)
                            
                            self.test_dataset.append(self.dataset_samples[idx])
                            self.test_seg.append((ck, cp))

               
                
            else:
                if args.sampling_scheme=="scheme1":
                    for ck in range(self.test_num_segment):
                        for idx in range(len(self.label_array)):
                            sample_label = self.label_array[idx]
                            sample = self.dataset_samples[idx]
                            self.test_indices[sample] = []
   
                            if isinstance(sample_label, list) and not args.approach=="video_classification":
                                test_seg_len = len(sample_label)//self.test_num_segment
                                if test_seg_len<1:
                                    pass
                                self.test_label_array.append(np.max(sample_label[ck*test_seg_len:(ck+1)*test_seg_len ]))
                                
                            else:
                                self.test_label_array.append(sample_label)

                            self.test_seg.append((ck))

                            self.test_dataset.append(self.dataset_samples[idx])
                        
                elif args.sampling_scheme =="scheme2":
                    
                    
                    for idx in range(len(self.label_array)):
                       
                        sample_label = self.label_array[idx]
                        sample = self.dataset_samples[idx]

                        self.test_indices[sample] = []
                            
                        chunks = int((len_vid(sample)- self.clip_len)//self.step_size) -1
                        for ck in range(chunks):
                        
                            self.test_label_array.append(sample_label)

                            self.test_seg.append((ck))

                            self.test_dataset.append(self.dataset_samples[idx])



        elif mode == 'inference':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        

            ])
            
            self.image_loader = get_image_loader()

         
        

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1
            # if args.masking:
            self.train_indices = {}
            mask_list = []
            sample = self.dataset_samples[index]
            # T H W C
            buffer= self.load_video(sample, sample_rate_scale=scale_t)
            
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer= self.load_video(sample, sample_rate_scale=scale_t)
                    if not sample in self.train_indices.keys():
                        self.train_indices[sample] = []

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                vid_list = []
                
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    if args.approach=='VAD':
                        label = max(self.label_array[index])#_*train_seg_len:(_+1)*train_seg_len ]
                    else:
                        label = self.label_array[index]
                    
                    # print("train label",index,label)
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                    vid_list.append(sample.split("/")[-1].split(".")[0] + "_"+ str(index))
                
                    if args.masking:
                        d , mask, _ = self.mask_generator((new_frames,None))
                        mask_list.append(mask)

                    # return frame_list, label_list, index_list, mask_list, {}
                if args.masking:    
                    return frame_list, label_list, index_list, mask_list,{}
                else:
                    return frame_list, label_list, index_list,{}
            else:
                buffer = self._aug_frame(buffer, args)
                if args.masking:
                    d , mask, _ = self.mask_generator((buffer,None))
                    # print(mask)
                    mask_list.append(mask)
            # labels = list(map(lambda x: max(x), self.label_array))

            if args.masking:
                _ , encoder_mask, _ = self.mask_generator((buffer, None))
                mask_list.append(mask)
                return buffer, self.label_array[index], index, mask_list, {}

            # else:
            return buffer, self.label_array[index], index,{}

        elif self.mode == 'validation':
            args = self.args
            # if args.masking:
            mask_list = []
            sample = self.val_dataset[index]
            chunk_nb, split_nb = self.val_seg[index]
            buffer= self.load_video(sample)

            while len(buffer) == 0:
                warnings.warn(
                    "video {}, temporal {}, spatial {} not found during testing"
                    .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.val_dataset[index]
                chunk_nb, split_nb = self.val_seg[index]
                buffer= self.load_video(sample)

            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.sparse_sample:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_start = chunk_nb
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start::self.test_num_segment,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start::self.test_num_segment, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]
            else:
                
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                      self.short_side_size) / (
                                          self.test_num_crop - 1)
                temporal_step = max(
                    1.0 * (buffer.shape[0] - self.clip_len) /
                    (self.test_num_segment - 1), 0)
                temporal_start = int(chunk_nb * temporal_step)
                spatial_start = int(split_nb * spatial_step)
                if buffer.shape[1] >= buffer.shape[2]:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :, :]
                else:
                    buffer = buffer[temporal_start:temporal_start +
                                    self.clip_len, :,
                                    spatial_start:spatial_start +
                                    self.short_side_size, :]

            buffer = self.data_transform(buffer)

            if args.masking:
                d , mask, _ = self.mask_generator((buffer,None))
                mask_list.append(mask)


            if args.masking:
                return buffer, self.val_label_array[index], sample.split(
                    "/")[-1].split(".")[0],mask_list, {}
            else:
                return buffer, self.val_label_array[index], sample.split(
                    "/")[-1].split(".")[0], {}
        
           

        elif self.mode == 'test':
            
            args = self.args
            sample = self.test_dataset[index]
            if sample not in self.test_indices.keys():
                self.test_indices[sample] = []
            
            
            buffer = self.load_video(sample)

            if args.test_randomization:
                chunk_nb,split_nb = self.test_seg[index]
            else:
                chunk_nb  = self.test_seg[index]
            
                

            mask_list = []

            if args.test_randomization:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {}, temporal {}, spatial {} not found during testing"
                        .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index]
                    chunk_nb, split_nb = self.test_seg[index]
                    buffer = self.load_video(sample)

            elif args.sampling_scheme=="scheme1":
                while len(buffer) == 0:
                    warnings.warn(
                        "video {}, temporal {}, spatial {} not found during testing"
                        .format(str(self.test_dataset[index]), chunk_nb))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index]
                    chunk_nb = self.test_seg[index]
                    buffer = self.load_video(sample)


            elif args.sampling_scheme=="scheme2":
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not found during testing"
                        .format(str(self.test_dataset[index])))
                    index = np.random.randint(self.__len__())
                    sample = self.test_dataset[index]
                    chunk_nb = self.test_seg[index]
                    
                    buffer = self.load_video2(sample)
                    self.test_indices[sample].append(indices)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            

            if args.test_randomization:

                if self.sparse_sample:
                    spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                        self.short_side_size) / (
                                            self.test_num_crop - 1)
                    temporal_start = chunk_nb
                    spatial_start = int(split_nb * spatial_step)
                    
                    
                    if buffer.shape[1] >= buffer.shape[2]:
                        buffer = buffer[temporal_start::self.test_num_segment,
                                        spatial_start:spatial_start +
                                        self.short_side_size, :, :]
                    else:
                        buffer = buffer[temporal_start::self.test_num_segment, :,
                                        spatial_start:spatial_start +
                                        self.short_side_size, :]
                else:
                    
                    spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                        self.short_side_size) / (
                                            self.test_num_crop - 1)
                    temporal_step = max(
                        1.0 * (buffer.shape[0] - self.clip_len) /
                        (self.test_num_segment - 1), 0)
                    temporal_start = int(chunk_nb * temporal_step)
                    spatial_start = int(split_nb * spatial_step)
                    
                    
                    if buffer.shape[1] >= buffer.shape[2]:
                        buffer = buffer[temporal_start:temporal_start +
                                        self.clip_len,
                                        spatial_start:spatial_start +
                                        self.short_side_size, :, :]
                    else:
                        buffer = buffer[temporal_start:temporal_start +
                                        self.clip_len, :,
                                        spatial_start:spatial_start +
                                        self.short_side_size, :]
            
            # else:
           

            elif args.sampling_scheme=="scheme1":
                if self.sparse_sample:
                    
                    temporal_start = chunk_nb
                    
                    buffer = buffer[temporal_start::self.test_num_segment, :,
                                    0:224, 0:224]
                    
            elif args.sampling_scheme=="scheme2":
                    
                    
                temporal_step = self.step_size
                
                temporal_start = int(chunk_nb * temporal_step)
                
                buffer = buffer[temporal_start:temporal_start +
                                self.clip_len,
                                    :, 0:224, 0:224]
             
                
        
            buffer = self.data_transform(buffer)
            
            if args.test_randomization:
                if args.masking:
                    d , mask, _ = self.mask_generator((buffer,None))
                    mask_list.append(mask)

                    return buffer, self.test_label_array[index], sample.split(
                        "/")[-1].split(".")[0], chunk_nb, split_nb, mask_list
                else:
                    return buffer, self.test_label_array[index], sample.split(
                        "/")[-1].split(".")[0], chunk_nb, split_nb, "".join([str(s) for s in range(temporal_start,temporal_start + self.clip_len)])
            elif args.sampling_scheme=="scheme1":
                if args.masking:
                    d , mask, _ = self.mask_generator((buffer,None))
                    mask_list.append(mask)

                    return buffer, self.test_label_array[index], sample.split(
                        "/")[-1].split(".")[0], chunk_nb, 0 ,mask_list
                else:
                    return buffer, self.test_label_array[index], sample.split(
                        "/")[-1].split(".")[0], chunk_nb, 0 , "".join([str(s) for s in range(temporal_start,temporal_start + self.clip_len)])
                
            elif args.sampling_scheme=="scheme2":
                if buffer.shape[1]==16:
                    if args.masking:
                        d , mask, _ = self.mask_generator((buffer,None))
                        mask_list.append(mask)

                        return buffer, self.test_label_array[index], sample.split(
                            "/")[-1].split(".")[0], chunk_nb, 0 ,mask_list
                    else:
                        return buffer, self.test_label_array[index], sample.split(
                            "/")[-1].split(".")[0], chunk_nb, 0,"".join([str(s) for s in range(temporal_start,temporal_start + self.clip_len)])
                else:
                    return None, None, None, None, 0

        elif self.mode == 'inference':
            
            args = self.args
            sample = self.inference_dataset[index]
            
            buffer = self.load_frame(sample)
            args = self.args

            # if args.masking:
            mask_list = []

            
            while len(buffer) == 0:
                warnings.warn(
                    "video {}, not found during testing"
                    .format(str(self.inference_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.inference_dataset[index]
                
                buffer = self.load_video2(sample)
            


            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            
            

            temporal_step = max(
                1.0 * (buffer.shape[0] - self.clip_len)/1 , 0)
            
            
            temporal_start = 0
            
            
            buffer = buffer[temporal_start:temporal_start +
                            self.clip_len,
                                :, 0:224, 0:224]
              
            buffer = self.data_transform(buffer)
            
            
            if args.masking:
                d , mask, _ = self.mask_generator((buffer,None))
                mask_list.append(mask)

                return buffer, self.inference_label_array[index], sample.split(
                    "/")[-1].split(".")[0], mask_list
            else:
                return buffer, self.inference_label_array[index], sample.split(
                    "/")[-1].split(".")[0]
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            # crop_size=224,
            crop_size=args.input_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)  # C T H W -> T C H W
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W -> C T H W

        return buffer

    def load_video(self, sample, sample_rate_scale=1):
        fname = sample

        try:
            vr = self.video_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)

        if self.mode == 'test':
            if self.sparse_sample:
                tick = length / float(self.num_segment)
                # tick =1 
                all_index = []
                for t_seg in range(self.test_num_segment):
                    tmp_index = [
                        int(t_seg * tick / self.test_num_segment + tick * x)
                        for x in range(self.num_segment)
                    ]
                    all_index.extend(tmp_index)
                all_index = list(np.sort(np.array(all_index)))
            else:
                all_index = [
                    x for x in range(0, length, self.frame_sample_rate)
                ]
                while len(all_index) < self.clip_len:
                    all_index.append(all_index[-1])

            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                if self.mode == 'validation':
                    end_idx = (converted_len + seg_len) // 2
                else:
                    end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def load_frame(self, sample, num_frames=16, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if self.mode == 'inference':
            tick = 1
            all_index = []
            # for t_seg in range(1):
            #     tmp_index = [
            #         int(t_seg * tick + tick * x)
            #         for x in range(self.num_segment)
            #     ]
            #     all_index.extend(tmp_index)
            all_index = os.listdir(fname)
            imgs = []
            for idx in all_index:
                frame_fname = os.path.join(fname,
                                           self.filename_tmpl.format(idx))
                img = self.image_loader(frame_fname)
                imgs.append(img)
            buffer = np.array(imgs)
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation':
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))
        elif num_frames > self.num_segment:
            if self.mode == 'validation':
                all_index = list(range(self.num_segment))
            else:
                all_index = list(
                    np.sort(
                        np.random.randint(num_frames, size=self.num_segment)))
        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index) + self.start_idx)
        imgs = []
        for idx in all_index:
            frame_fname = os.path.join(fname, self.filename_tmpl.format(idx))
            img = self.image_loader(frame_fname)
            imgs.append(img)
        buffer = np.array(imgs)
        return buffer


    def load_video2(self, sample):
        fname = sample

        try:
            vr = self.image_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)

        if self.mode == 'test':
            
            all_index = [
                x for x in range(0, length)
            ]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])

            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

class RawFrameClsDataset(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(self,
                 anno_path,
                 data_root,
                 mode='train',
                 clip_len=8,
                 crop_size=224,
                 short_side_size=256,
                 new_height=256,
                 new_width=340,
                 keep_aspect_ratio=True,
                 num_segment=1,
                 num_crop=1,
                 test_num_segment=1,
                 test_num_crop=2,
                 filename_tmpl='img_{:05}.jpg',
                 start_idx=1,
                 args=None):
        self.anno_path = anno_path
        self.data_root = data_root
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.start_idx = start_idx
        self.args = args
        self.aug = False
        self.rand_erase = False

        # self.frame_level_anno_path = "/home/mayuna/scratch/videos/original_dataset_4.csv"

        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self.image_loader = get_image_loader()

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        # self.frame_level_df = pd.read_csv(self.frame_level_anno_path, delimiter=',')
        #read the video segments 
        # self.frame_level_df = self.frame_level_df.dropna()
        # self.frame_level_df = self.frame_level_df[['File_name','video_names', '#frames', 'vid_num','pt_num', 'frames_with_cancer', 'num_malignancy', 'annotation', 'ambiguous']]

        # self.frame_level_df 

        self.dataset_samples = list(
            cleaned[0].apply(lambda row: os.path.join(self.data_root, row)))
        # print(cleaned)
        self.total_frames = list(cleaned.values[:, 1])
        self.label_array = list(cleaned.values[:, -1])
        # print(self.label_array)


        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(
                    self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(
                    size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(
                    size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_total_frames = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        self.test_seg.append((ck, cp))
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_total_frames.append(self.total_frames[idx])
                        self.test_label_array.append(self.label_array[idx])

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args
            scale_t = 1

            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(
                sample, total_frame, sample_rate_scale=scale_t)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during training".format(
                            sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    total_frame = self.total_frames[index]
                    buffer = self.load_frame(
                        sample, total_frame, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)

            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            total_frame = self.total_frames[index]
            buffer = self.load_frame(sample, total_frame)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn(
                        "video {} not correctly loaded during validation".
                        format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.load_frame(sample, total_frame)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split(
                "/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            total_frame = self.test_total_frames[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.load_frame(sample, total_frame)

            while len(buffer) == 0:
                warnings.warn(
                    "video {}, temporal {}, spatial {} not found during testing"
                    .format(str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                total_frame = self.test_total_frames[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.load_frame(sample, total_frame)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) -
                                  self.short_side_size) / (
                                      self.test_num_crop - 1)
            temporal_start = chunk_nb
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start::self.test_num_segment,
                                spatial_start:spatial_start +
                                self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start::self.test_num_segment, :,
                                spatial_start:spatial_start +
                                self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split(
                "/")[-1].split(".")[0], chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(self, buffer, args):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_frame(self, sample, num_frames, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if self.mode == 'test':
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index) + self.start_idx))
            imgs = []
            for idx in all_index:
                frame_fname = os.path.join(fname,
                                           self.filename_tmpl.format(idx))
                img = self.image_loader(frame_fname)
                imgs.append(img)
            buffer = np.array(imgs)
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation':
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(
                        list(range(self.num_segment)), average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))
        elif num_frames > self.num_segment:
            if self.mode == 'validation':
                all_index = list(range(self.num_segment))
            else:
                all_index = list(
                    np.sort(
                        np.random.randint(num_frames, size=self.num_segment)))
        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index) + self.start_idx)
        imgs = []
        for idx in all_index:
            frame_fname = os.path.join(fname, self.filename_tmpl.format(idx))
            img = self.image_loader(frame_fname)
            imgs.append(img)
        buffer = np.array(imgs)
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift else video_transforms.random_resized_crop)
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale)
        frames, _ = video_transforms.uniform_crop(frames, crop_size,
                                                  spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
