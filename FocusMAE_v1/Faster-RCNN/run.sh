python test.py \
    --img_dir='path_to_img_dir' \
    --meta_file='json file with image name as key and a dictionary containing boxes and labels as values' \  # boxex will be list of list with coordinates as [0,0,0,0] for all images
    --label_file='txt file with image name and label' \  # image_name,label 
    --out_dir='path_to_out_dir' \
    --pretrained_weights='path_to_pretrained_fasterrcnn_weights'