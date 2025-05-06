# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os

from .datasets import RawFrameClsDataset, VideoClsDataset,VideoClsDataset_v2
from .pretrain_datasets import (  # noqa: F401
    DataAugmentationForVideoMAEv2,DataAugmentationForCandidateRegionVideomae, HybridVideoMAE, VideoMAE,DataAugmentationForFineTune, VideoMAE_Inference, Candidate_ROI_Videomae
)


def build_pretraining_dataset(args , test_mode):
    transform = DataAugmentationForCandidateRegionVideomae(args, test_mode)
    if not test_mode:
        transform = DataAugmentationForCandidateRegionVideomae(args, test_mode)
        # dataset = VideoMAE(
        #     root=args.data_root,
        #     setting=args.data_path,
        #     train=True,
        #     test_mode=test_mode,
        #     name_pattern=args.fname_tmpl,
        #     video_ext='mp4',
        #     is_color=True,
        #     modality='rgb',
        #     num_segments=1,
        #     num_crop=1,
        #     new_length=args.num_frames,
        #     new_step=args.sampling_rate,
        #     transform=transform,
        #     temporal_jitter=False,
        #     lazy_init=False,
        #     num_sample=args.num_sample)
        dataset = Candidate_ROI_Videomae(
            root=args.data_root,
            json_file_path=args.json_file_path,
            dir_path=args.images_folder_path,
            setting=args.data_path,
            train=True,
            test_mode=False,
            name_pattern=args.fname_tmpl,
            video_ext='mp4',
            is_color=True,
            modality='rgb',
            new_step=args.sampling_rate,
            new_length=4,
            num_sample = args.num_sample, 
            num_segments = args.num_segments,
            transform=transform,
            temporal_jitter=False,
            lazy_init=False
            )
    else:
        print("loading video inference")
        transform = DataAugmentationForCandidateRegionVideomae(args, test_mode)
        # dataset = VideoMAE_Inference(
        #     root=args.data_root,
        #     setting=args.data_root_inference,
        #     train=False,
        #     test_mode=True,
        #     name_pattern='{:05}.jpg',
        #     video_ext='mp4',
        #     is_color=True,
        #     modality='rgb',
        #     new_length=4,
        #     new_step=1,
        #     transform=transform,
        #     temporal_jitter=False,
        #     lazy_init=False,
        # )
        dataset = Candidate_ROI_Videomae(
            root=args.data_root,
            setting=args.data_root_inference,
            train=False,
            test_mode=False,
            name_pattern=args.fname_tmpl,
            video_ext='mp4',
            is_color=True,
            modality='rgb',
            new_step=args.sampling_rate,
            new_length = 4,
            transform=transform,
            temporal_jitter=False,
            lazy_init=False,
            )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args,mode="train"):
    if is_train:
        mode = 'train'
        anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode:
        mode = 'test'
        anno_path = os.path.join(args.data_path, 'test.csv')
    else:
        mode = 'validation'
        anno_path = os.path.join(args.data_path, 'val.csv')

    if args.data_set == 'Kinetics-400':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
            # nb_classes = 2

        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 2

    elif args.data_set == 'Kinetics-600':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 600

    elif args.data_set == 'Kinetics-700':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 700

    elif args.data_set == 'Kinetics-710':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 710

    elif args.data_set == 'SSV2':
        dataset = RawFrameClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.fname_tmpl,
            start_idx=args.start_idx,
            args=args)

        nb_classes = 174

    elif args.data_set == 'UCF101':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101

    elif args.data_set == 'HMDB51':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51

    elif args.data_set == 'Diving48':
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    elif args.data_set == 'MIT':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = 339

    elif args.data_set == 'GBC_Net':
        
        if args.masking:
            transform = DataAugmentationForFineTune(args)
        else:
            transform = None
        if not args.sparse_sample:
            dataset = VideoClsDataset_v2(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=args.num_segment,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 2,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                sparse_sample=args.sparse,
                transform  = transform, 
                step_size= args.step_size,
                args=args)
            # nb_classes = 2

        else:
            dataset = VideoClsDataset_v2(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=args.num_segment,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=224,
                new_width=224,
                sparse_sample=args.sparse,
                transform = transform,
                step_size= args.step_size,
                args=args)
        nb_classes = 2


    
    elif args.data_set == 'CT_Data':
        dataset = RawFrameClsDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=16,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=224,
            new_width=224,
            filename_tmpl=args.fname_tmpl,
            start_idx=args.start_idx,
            args=args)

        nb_classes = 2

    else:
        raise NotImplementedError('Unsupported Dataset')

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
