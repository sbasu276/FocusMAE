# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np

import random 
class Cell():

    def __init__(self, num_masks, num_patches):
        self.num_masks = num_masks
        self.num_patches = num_patches
        self.size = num_masks + num_patches
        self.queue = np.hstack([np.ones(num_masks), np.zeros(num_patches)])
        self.queue_ptr = 0

    def set_ptr(self, pos=-1):
        self.queue_ptr = np.random.randint(self.size) if pos < 0 else pos

    def get_cell(self):
        cell_idx = (np.arange(self.size) + self.queue_ptr) % self.size
        return self.queue[cell_idx]

    def run_cell(self):
        self.queue_ptr += 1


class RandomMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 3

        self.frames, self.height, self.width = input_size

        self.num_patches = self.frames * self.height * self.width  # 8x14x14
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Mask: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask)
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196*8]


class TubeMaskingGenerator:

    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        return mask  # [196*8]
    

class OursMaskingGeneratorv1:


    def __init__(self, image_size, roi_boxes , patch_size = (16,16,2), mask_ratio=0.8, representative_frame=0,inflation_ratio=1.5):
        
        self.frames, self.height, self.width = image_size
        self.patch_size = patch_size[0]
        self.roi_box_list = []
        self.tube = patch_size[2]
        self.input_size = self.height
        for roi_box in roi_boxes:
            rb = []
            if len(roi_box)>0:
                for r in roi_box:
                    r = self.roi_region_inflater(r, ratio=inflation_ratio)
                    rb.append(r)
                   
            else:#incase of no bounding box at all for usg we can focus on the centre region of the image due to balg regions in ultrasound images
                rb.append([self.height//4, self.width//4, self.height*3//4, self.width*3//4])

            self.roi_box_list.append(rb)
        self.num_patch_per_frames = (self.height//patch_size[0]) * (self.width//patch_size[1]) 
        self.total_patches = (self.frames//patch_size[2]) * self.num_patch_per_frames
        
        self.patch_not_in_rois = np.zeros(self.num_patch_per_frames)  #we are making a mask only for the first frame and then repeat it 
            
        for y in range(self.num_patch_per_frames): 

            self.patch_not_in_rois[y] = self.is_patch_in_roi(y, self.roi_box_list[representative_frame])
                

        self.num_masks = sum(self.patch_not_in_rois)

    def roi_region_inflater(self, boxes, ratio = 1.1):
        x1 ,y1, x2, y2 = boxes
        x4 = x1 + (x2-x1)*(1+ (ratio-1)*0.5)
        y4 = y1 + (y2-y1)*(1+ (ratio-1)*0.5)
        x3 = x1 - (x2 - x1)*(ratio-1)*0.5
        y3 = y1 - (y2 - y1)*(ratio-1)*0.5

        return x3,y3,x4,y4
    
    def is_patch_in_roi(self, patch_number, roi_boxes):
        x1 ,y1 = max((patch_number//(self.input_size//self.patch_size))*self.patch_size - self.patch_size//2 , 0) , max( (patch_number%(self.input_size//self.patch_size)) *self.patch_size - self.patch_size//2 , 0)
        for roi_box in roi_boxes:
            x1r,y1r,x2r,y2r = roi_box
            
            x1r = int(x1r*224/360.0)
            y1r = int(y1r*224/480.0)
            x2r = int(x2r*224/360.0)
            y2r = int(y2r*224/480.)

            if x1 < min(x1r, x2r) or x1 > max(x1r, x2r):
                continue
            elif y1 < min(y1r, y2r) or  y1 > max(y1r, y2r):
                continue
            else:
                return 1
        return 0

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
             self.total_patches,self.num_masks)
        return repr_str

    def __call__(self):
       
       
        mask_per_frame = self.patch_not_in_rois

        mask = np.tile(mask_per_frame, (self.frames//self.tube, 1))

        return  mask# [196*8]


class OursMaskingGeneratorv3:


    def __init__(self, image_size, roi_boxes , patch_size = (16,16,2), mask_ratio=0.8, representative_frame=0):
        
        self.frames, self.height, self.width = image_size

        self.roi_box_list = []
        self.tube = patch_size[2]
        for roi_box in roi_boxes:
            rb = []
            if len(roi_box)>0:
                for r in roi_box:
                    r = self.roi_region_inflater(r)
                    rb.append(r)
            else:
                rb.append([self.height//4, self.width//4, self.height*3//4, self.width*3//4])

            self.roi_box_list.append(rb)
        self.num_patch_per_frames = (self.height//patch_size[0]) * (self.width//patch_size[1]) 
        self.total_patches = (self.frames//patch_size[2]) * self.num_patch_per_frames
        self.patch_not_in_rois = np.zeros(self.num_patch_per_frames)  #we are making a mask only for the first frame and then repeat it 

        set_of_boxes = self.roi_box_list[0] +  self.roi_box_list[len(self.roi_box_list)//2] + self.roi_box_list[-1]

        for y in range(self.num_patch_per_frames): 

            self.patch_not_in_rois[y] = self.is_patch_in_roi(y, set_of_boxes)

        self.num_masks = int(mask_ratio * sum(self.patch_not_in_rois))

    def roi_region_inflater(self, boxes, ratio = 1.1):
        x1 ,y1, x2, y2 = boxes
        x4 = x1 + (x2-x1)*(1+ (ratio-1)*0.5)
        y4 = y1 + (y2-y1)*(1+ (ratio-1)*0.5)
        x3 = x1 - (x2 - x1)*(ratio-1)*0.5
        y3 = y1 - (y2 - y1)*(ratio-1)*0.5

        return x3,y3,x4,y4
    
    def is_patch_in_roi(self, patch_number, roi_boxes):
        x1 ,y1 = max((patch_number%14)*16 -8 , 0) , max( (patch_number//14) *16 -8 , 0)
        for roi_box in roi_boxes:
            x1r,y1r,x2r,y2r = roi_box
        
            x1r = int(x1r*224/360.0)
            y1r = int(y1r*224/480.0)
            x2r = int(x2r*224/360.0)
            y2r = int(y2r*224/480.)

            if x1 < min(x1r, x2r) or x1 > max(x1r, x2r):
                continue
            elif y1 < min(y1r, y2r) or  y1 > max(y1r, y2r):
                continue
            else:
                return 1
        return 0

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
             self.total_patches,self.num_masks)
        return repr_str

    def __call__(self):
       
        indices = [ind for ind, ele in enumerate(self.patch_not_in_rois) if ele == 1]

        mask_per_frame = self.patch_not_in_rois
        
        mask_one_roi = random.choices(indices, k=self.num_masks)

            
        for x in mask_one_roi:
            mask_per_frame[x] = 1 - mask_per_frame[x]

        mask = np.tile(mask_per_frame, (self.frames//self.tube, 1))

        return  mask# [196*8]


class OursMaskingGeneratorv2:


    def __init__(self, image_size, roi_boxes , patch_size = (16,16,2), mask_ratio=0.8, representative_frame=0):
        
        self.frames, self.height, self.width = image_size

        self.roi_box_list = []
        self.tube = patch_size[2]
        for roi_box in roi_boxes:
            rb = []
            if len(roi_box)>0:
                for r in roi_box:
                    r = self.roi_region_inflater(r)
                    rb.append(r)
            else:
                rb.append([self.height//4, self.width//4, self.height*3//4, self.width*3//4])

            self.roi_box_list.append(rb)
        self.num_patch_per_frames = (self.height//patch_size[0]) * (self.width//patch_size[1]) 
        self.total_patches = (self.frames//patch_size[2]) * self.num_patch_per_frames

        self.patch_not_in_rois = np.zeros(self.num_patch_per_frames)  #we are making a mask only for the first frame and then repeat it 

            
        for y in range(self.num_patch_per_frames): 

            self.patch_not_in_rois[y] = self.is_patch_in_roi(y, self.roi_box_list[representative_frame])

        self.num_masks = int(mask_ratio * sum(self.patch_not_in_rois))

    def roi_region_inflater(self, boxes, ratio = 1.1):
        x1 ,y1, x2, y2 = boxes
        x4 = x1 + (x2-x1)*(1+ (ratio-1)*0.5)
        y4 = y1 + (y2-y1)*(1+ (ratio-1)*0.5)
        x3 = x1 - (x2 - x1)*(ratio-1)*0.5
        y3 = y1 - (y2 - y1)*(ratio-1)*0.5

        return x3,y3,x4,y4
    
    def is_patch_in_roi(self, patch_number, roi_boxes):
        #if centree of patch in roi region we count it as in (rasterization) 
        
        x1 ,y1 = max((patch_number%14)*16 -8 , 0) , max( (patch_number//14) *16 -8 , 0)
        for roi_box in roi_boxes:
            x1r,y1r,x2r,y2r = roi_box
        
            x1r = int(x1r*224/360.0)
            y1r = int(y1r*224/480.0)
            x2r = int(x2r*224/360.0)
            y2r = int(y2r*224/480.)

            if x1 < min(x1r, x2r) or x1 > max(x1r, x2r):
                continue
            elif y1 < min(y1r, y2r) or  y1 > max(y1r, y2r):
                continue
            else:
                return 1
        return 0

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
             self.total_patches,self.num_masks)
        return repr_str

    def __call__(self):
       
        indices = [ind for ind, ele in enumerate(self.patch_not_in_rois) if ele == 1]
       
        mask_per_frame = self.patch_not_in_rois
        
        mask_one_roi = random.choices(indices, k=self.num_masks)

            
        for x in mask_one_roi:
            mask_per_frame[x] = 1 - mask_per_frame[x]

        # mask = np.tile(mask_ , (self.frames//self.tube ,self.num_patch_per_frames))
        mask = np.tile(mask_per_frame, (self.frames//self.tube, 1))
        # print(mask.shape)

        return  mask# [196*8]


class RunningCellMaskingGenerator:

    def __init__(self, input_size, mask_ratio=0.5):
        self.frames, self.height, self.width = input_size
        self.mask_ratio = mask_ratio

        num_masks_per_cell = int(4 * self.mask_ratio)
        assert 0 < num_masks_per_cell < 4
        num_patches_per_cell = 4 - num_masks_per_cell

        self.cell = Cell(num_masks_per_cell, num_patches_per_cell)
        self.cell_size = self.cell.size

        mask_list = []
        for ptr_pos in range(self.cell_size):
            self.cell.set_ptr(ptr_pos)
            mask = []
            for _ in range(self.frames):
                self.cell.run_cell()
                mask_unit = self.cell.get_cell().reshape(2, 2)
                mask_map = np.tile(mask_unit,
                                   [self.height // 2, self.width // 2])
                mask.append(mask_map.flatten())
            mask = np.stack(mask, axis=0)
            mask_list.append(mask)
        self.all_mask_maps = np.stack(mask_list, axis=0)

    def __repr__(self):
        repr_str = f"Running Cell Masking with mask ratio {self.mask_ratio}"
        return repr_str

    def __call__(self):
        mask = self.all_mask_maps[np.random.randint(self.cell_size)]
        return np.copy(mask)

class EmptyMask:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.total_masks = int(mask_ratio * self.total_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        return []