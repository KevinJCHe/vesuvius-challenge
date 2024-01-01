import itertools
import os

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import scipy.stats as st

from albumentations.pytorch import ToTensorV2
from tap import Tap
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import cv2
import albumentations as A

from model import RegressionPLModel

iteration = 2
# ROOT_DIR = "C:\\Users/Epicurus"
ROOT_DIR =  "/mnt/c/Users/Epicurus"
DATA_PATH = f"{ROOT_DIR}/Documents/Scrolls-Project/data/fragments"
MODEL_PATH = f'./models/iteration2/I3D_iteration2-v9.ckpt'
OUTPUT_PATH = f"{ROOT_DIR}/Dropbox/Scrolls and Void/Kevin/predictions/GP_Iteration_{iteration}"

reverse_fragments = {
    '20231012173610',
    '20231007101615',
    '20231022170900',
    '20231011111857'
}

class ARGS:
    # TODO: make this into argument parsers
    segment_path = DATA_PATH
    model_path = MODEL_PATH
    out_path = OUTPUT_PATH
    stride = 8
    start_idx = 15
    in_chans = 30 # 65
    tile_size = 64
    train_batch_size = 128
    valid_batch_size = 128
    num_workers = 1

def crop(img, height_window=None, width_window=None):
    if height_window is not None:
        h_start, h_end = height_window
        img = img[h_start:h_end, :]
    if width_window is not None:
        e_start, e_end = width_window
        img = img[:, e_start:e_end]
    return img

def read_image_mask(fragment_id, start_depth=18, end_depth=38, height_window=None, width_window=None):
    
    fragment_mask = cv2.imread(f"{ARGS.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
    fragment_mask = crop(fragment_mask, height_window, width_window)

    pad0 = (ARGS.tile_size - fragment_mask.shape[0] % ARGS.tile_size)
    pad1 = (ARGS.tile_size - fragment_mask.shape[1] % ARGS.tile_size)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    kernel = np.ones((16,16), np.uint8)
    fragment_mask = cv2.erode(fragment_mask, kernel, iterations=1)
    
    if np.all(fragment_mask == 0):
        print('Empty fragment mask')
        return [], None

    images = []
    for d in tqdm(range(start_depth, end_depth), desc='Reading images...'):
        image = cv2.imread(f"{ARGS.segment_path}/{fragment_id}/layers/{d:02}.tif", 0)
        image = crop(image, height_window, width_window)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        image = np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)

    # REVERSE THE SLICES (FOR SOME FRAGMENTS, THE RESULTS ARE BETTER WHEN REVERSED)
    if fragment_id in reverse_fragments:
        print('REVERSING THE SLICES')
        images=images[:,:,::-1]
        
    return images, fragment_mask

def get_img_splits(*args, batch_size=ARGS.valid_batch_size):
    voxel_tiles = []
    xyxys = []
    voxels, fragment_mask = read_image_mask(*args)
    if fragment_mask is None:
        return
    
    x1_list = list(range(0, voxels.shape[1]-ARGS.tile_size+1, ARGS.stride))
    y1_list = list(range(0, voxels.shape[0]-ARGS.tile_size+1, ARGS.stride))
    for y1, x1 in tqdm(list(itertools.product(y1_list, x1_list))):
        y2 = y1 + ARGS.tile_size
        x2 = x1 + ARGS.tile_size
        if not np.any(fragment_mask[y1:y2, x1:x2]==0):
            voxel_tiles.append(voxels[y1:y2, x1:x2])
            xyxys.append([x1, y1, x2, y2])

    test_dataset = CustomDatasetTest(
        voxel_tiles, 
        np.stack(xyxys), 
        args, 
        transform=A.Compose([
            A.Resize(ARGS.tile_size, ARGS.tile_size),
            A.Normalize(
                mean= [0] * ARGS.in_chans,
                std= [1] * ARGS.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ])
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ARGS.num_workers, 
        pin_memory=True, 
        drop_last=False
    )
    return test_loader, np.stack(xyxys), (voxels.shape[0],voxels.shape[1]), fragment_mask


class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    model.eval()
    kernel=gkern(ARGS.tile_size,1)
    kernel=kernel/kernel.max()
    for images,xys in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(
                F.interpolate(
                    y_preds[i].unsqueeze(0).float(),
                    scale_factor=4,
                    mode='bilinear'
                ).squeeze(0).squeeze(0).numpy(),
                kernel
            )
            mask_count[y1:y2, x1:x2] += np.ones((ARGS.tile_size, ARGS.tile_size))

    mask_pred /= mask_count
    return mask_pred

def get_section_intervals(n, section_size, padding):
    # why add padding? So sections have overlaps
    result = []
    for i in range(0, n, section_size):
        start = max(0, i - padding)
        end = min(n, i + section_size + padding)
        if end - start < padding: break
        result.append([start, end])
    return result

def infer(
    fragment_id, 
    model,
    batch_size=128, # 8GB GPU RAM
    section_size=10000, 
    extra_padding=100
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = cv2.imread(f"{ARGS.segment_path}/{fragment_id}/{fragment_id}_mask.png", 0)
    height, width = mask.shape
    height_subsections = get_section_intervals(height, section_size, extra_padding)
    width_subsections = get_section_intervals(width, section_size, extra_padding)

    result = np.empty((height, width)).astype(np.uint8)
    for height_start, height_end in height_subsections:
        for width_start, width_end in width_subsections:

            output = get_img_splits(
                fragment_id,
                ARGS.start_idx,
                ARGS.start_idx+ARGS.in_chans,
                (height_start, height_end), # height window
                (width_start, width_end), # width window
                batch_size=batch_size
            )

            if output is None:
                print(fragment_id, height_start, height_end, width_start, width_end)
                continue

            test_loader, test_xyxz, test_shape, fragment_mask = output
            pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
            pred = np.clip(np.nan_to_num(pred),a_min=0,a_max=1)
            pred/= pred.max()
            pred = (pred*255).astype(np.uint8)

            # there are padding at the ends (at the right and at the bottom), ignore them
            pred = pred[:height_end - height_start, :width_end - width_start]
            # remove the extra paddings
            h_start, h_end, w_start, w_end = height_start, height_end, width_start, width_end
            if h_start != 0:
                pred = pred[extra_padding:]
                h_start += extra_padding
            if h_end != height:
                pred = pred[:-extra_padding] 
                h_end -= extra_padding
            if w_start != 0:
                pred = pred[:, extra_padding:]
                w_start += extra_padding
            if w_end != width:
                pred = pred[:, :-extra_padding]
                w_end -= extra_padding
            # combine mask subsection to the overall output
            result[h_start:h_end, w_start:w_end] = pred
    return result


def main():
    model=RegressionPLModel.load_from_checkpoint(ARGS.model_path,strict=False)
    model.cuda()
    model.eval()
    
    fragment_ids = [
        '20230702185753', # shape=(13513, 17381)
        '20231012173610', # shape=(31209, 10250)
        '20231012184422', # shape=(11882, 23364)
        '20231022170900', # shape=(34588, 7676)
        '20231031143852', # shape=(10620, 13577)
        '20231106155351', # shape=(10636, 15738)
        '20231005123336', # shape=(10939, 29872)
        '20231012085431', # shape=(10187, 8565)
        '20231024093300', # shape=(13730, 5637)
        '20231007101615',
        '20231016151001',
    ]

    
    """
    TESTING 

    20230519195952. shape = (1800, 2270)
    stride 2, batch size 128 - Infer Time: 20min
    stride 2, batch size 128, fp16 (model.half() and model(images.half())) - Infer Time: 19min
    stride 2, batch size 256 - Infer Time: 20min
    stride 4, batch size 256 - Infer Time: 5min <- NO DROP IN QUALITY
    stride 8, batch size 256 - Infer Time: 2min <- NO DROP IN QUALITY
    """
    # fragment_ids = ['20230519195952']
    
    for fragment_id in fragment_ids:
        result = infer(fragment_id, model)
        result = Image.fromarray(result)
        result.save(f'{ARGS.out_path}/{fragment_id}.png')

if __name__ =='__main__':
    main()