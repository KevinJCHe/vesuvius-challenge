import itertools
import os
import random

from albumentations.pytorch import ToTensorV2
from glob import glob
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
import cv2
import imageio
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model import RegressionPLModel
from inference import infer, get_img_splits, predict_fn

# CONFIG
iteration = 1

big_tile_size = 256
small_tile_size = 64
stride = big_tile_size // 8 # 32
start_slice = 15
end_slice = 45
max_epochs = 10
num_workers = 16

input_channels = end_slice - start_slice

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PATHS
# ROOT_DIR = "C:\\Users/Epicurus"
ROOT_DIR =  "/mnt/c/Users/Epicurus"
LABEL_PATH = f"{ROOT_DIR}/Dropbox/Scrolls and Void/Babak/Iteration_{iteration}"
DATA_PATH = f"{ROOT_DIR}/Documents/Scrolls-Project/data/fragments"
# MODEL_PATH = './models/BASE_MODEL_valid_20230827161847_0_fr_i3depoch=7.ckpt'
# MODEL_PATH = f'./models/archive/iteration1-2023-12-23-p1/I3D_iteration1-v9.ckpt'
epoch_offset = 0
MODEL_PATH = f'./models/iteration1/I3D_iteration1-v1.ckpt'
OUTPUT_PATH = f"{ROOT_DIR}/Dropbox/Scrolls and Void/Kevin/predictions/GP_Iteration_{iteration+1}/Training"

rotate_fragments = {
    '20230702185753': 90,
    '20230820203112': 90,
    '20230827161847': 90,
    '20230904135535': 90,
    '20231012184427': 90, # TO ASK: where did 20231012184427 go?
}

reverse_fragments = {
    '20231012173610',
    '20231007101615',
    '20231022170900',
    '20231011111857'
}

train_aug_list = [
    # A.RandomResizedCrop(size, size, scale=(0.85, 1.0)),
    A.Resize(small_tile_size, small_tile_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.6),
    A.RandomBrightnessContrast(p=0.75),
    A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
    A.OneOf([A.GaussNoise(var_limit=[10, 50]), A.GaussianBlur(), A.MotionBlur(),], p=0.4),
    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.CoarseDropout(max_holes=2, max_width=int(small_tile_size * 0.2), max_height=int(small_tile_size * 0.2), mask_fill_value=0, p=0.5),
    # A.Cutout(max_h_size=int(size * 0.6), max_w_size=int(size * 0.6), num_holes=1, p=1.0),
    A.Normalize(mean=[0] * input_channels, std=[1] * input_channels),
    ToTensorV2(transpose_mask=True),
]

def get_data(df, write_fragment_chunks=True):
    """
    Args:
        write_fragment_chunks: If True, write the small tiles on disk. If False, skip reading 
            the fragment data altogether, assuming the small tiles are already written for ALL
            fragments in df
    """
    data_paths = []
    labels = []
    for fragment_id, fragment_group in df.groupby('fragment_id'):
        # get the label mask
        masks = [cv2.imread(f"{LABEL_PATH}/{mask_filename}", 0) for mask_filename in fragment_group['mask_filename'].unique()]
        label_mask = np.stack(masks, axis=2).max(axis=2)
        label_mask = label_mask.astype('float32')
        label_mask /= 255
        assert len(np.unique(label_mask.flatten())) == 2 # values are either 0 or 1, no in-between

        # get the fragment mask
        fragment_mask = cv2.imread(f"{DATA_PATH}/{fragment_id}/{fragment_id}_mask.png", 0)
        if fragment_id in rotate_fragments: fragment_mask = np.rot90(fragment_mask)

        # get the coords and small label tiles
        label, coords = get_label_coords(label_mask, fragment_mask, fragment_id)
        unique_coords = list(set(map(tuple, coords)))
        
        if write_fragment_chunks:
            # get the fragment data
            data = get_fragment_data(unique_coords, fragment_id)
            print(f'Shape of unique data for fragment id {fragment_id}: {data.shape}')
        else:
            print(f'Number of unique small tiles for fragment id {fragment_id}: {len(unique_coords)}')

        # cannot load all the data in-memory, we must store the data on disk and read when we need it
        for i in range(len(coords)):
            filename = f'{fragment_id}_{"_".join(map(str, coords[i]))}' 
            #filepath = f'./data/Iteration_1/{filename}.tif'
            filepath = f'./data/Iteration_{iteration+1}/{filename}.tif'
            if write_fragment_chunks and not os.path.exists(filepath): 
                # NOTE: There are many duplicate small tiles (due to many overlapping big tiles)        16 to be exact, based on 256 big tile size, 64 samll tile size, 32 stride
                # no need to write the same tiles over and over again
                idx = unique_coords.index(coords[i])
                imageio.mimwrite(filepath, data[idx])
                # TODO: might as well write the labels too
            data_paths.append(filepath)
            labels.append(label[i])
    data_paths = np.array(data_paths)
    labels = np.array(labels)
    return data_paths, labels


def get_label_coords(label_mask, fragment_mask, fragment_id):
    height, width = label_mask.shape
    # unit test
    h, w = fragment_mask.shape
    assert (h, w) == (height, width)

    coords, label = [], []
    ys = range(0, height-big_tile_size+1, stride)
    xs = range(0, width-big_tile_size+1, stride)
    for y, x in tqdm(list(itertools.product(ys, xs)), desc=f'Getting segment coordinates for fragment id {fragment_id}...'):
        if np.any(fragment_mask[y:y+big_tile_size, x:x+big_tile_size] == 0): continue
        if np.all(label_mask[y:y+big_tile_size, x:x+big_tile_size] == 0): continue
            
        y_steps = range(0, big_tile_size, small_tile_size)
        x_steps = range(0, big_tile_size, small_tile_size)
        for y_step, x_step in itertools.product(y_steps, x_steps):
            y_start, x_start = y+y_step, x+x_step
            y_end, x_end = y_start+small_tile_size, x_start+small_tile_size
            mask = label_mask[y_start:y_end, x_start:x_end]
            label.append([mask])
            coords.append([y_start, y_end, x_start, x_end])
    label = np.array(label)
    # label.shape = (number of tiles, 1, height, width)

    # ensure shape matches with youssef's
    label = np.moveaxis(np.array(label), 1, -1) 
    # label.shape = (number of tiles, height, width, 1)

    return label, coords

def get_fragment_data(unique_coords, fragment_id):
    data = [[] for _ in range(len(unique_coords))]
    for slice_i in tqdm(range(start_slice, end_slice), desc=f"Reading fragment {fragment_id} images..."):
        image = cv2.imread(f"{DATA_PATH}/{fragment_id}/layers/{slice_i:02}.tif", 0)
        image = np.clip(image,0,200)
        if fragment_id in rotate_fragments: image = np.rot90(image)
        for tile_i, (y_start, y_end, x_start, x_end) in enumerate(unique_coords):
            data[tile_i].append(image[y_start:y_end, x_start:x_end])
    data = np.array(data)
    # data.shape = (number of tiles, depth, height, width)

    # ensure shape matches with youssef's
    data = np.moveaxis(np.array(data), 1, -1) 
    # data.shape = (number of tiles, height, width, depth)

    # reverse the fragment slices
    if fragment_id in reverse_fragments:
        print(f'Reversing the fragment slices for {fragment_id}')
        data = data[:,:,::-1]

    return data


class CustomDataset(Dataset):
    def __init__(self, data_paths, labels, transform):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.input_channels = input_channels

    def __len__(self):
        return len(self.data_paths)
    
    def youssef_augment(self, voxel):
        voxel_tmp = np.zeros_like(voxel)
        cropping_num = random.randint(22, 30)
        start_idx = random.randint(0, self.input_channels - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        start_paste_idx = random.randint(0, self.input_channels - cropping_num) 
        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)
        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]
        voxel_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = voxel[..., crop_indices]
        if random.random() > 0.4:
            voxel_tmp[..., temporal_random_cutout_idx] = 0
        voxel = voxel_tmp
        return voxel

    def __getitem__(self, idx):
        voxel = np.array(imageio.mimread(self.data_paths[idx]))
        label = self.labels[idx]
        voxel = self.youssef_augment(voxel) 
        data = self.transform(image=voxel, mask=label)
        voxel = data['image'].unsqueeze(0)
        label = data['mask']
        label = F.interpolate(label.unsqueeze(0), (small_tile_size//4, small_tile_size//4)).squeeze(0)
        return voxel, label
    

class SmallCustomCallback(pl.Callback):
    """Save the prediction output for a small fragment at end of each epoch"""
    def __init__(self, dirpath, fragment_id='20230519195952'):
        self.path = f'{dirpath}/preds/{fragment_id}'
        self.fragment_id = fragment_id
        self.loader, self.xyxz, self.shape, _ = get_img_splits(
            fragment_id,
            start_slice,
            end_slice,
            None, # height window
            None, # width window
            batch_size=128,
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thresholds = np.arange(0.3, 0.9, 0.1)
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            for thresh in self.thresholds:
                os.makedirs(f'{self.path}/thresholding/thresh_{thresh:.1}')

    def on_train_epoch_end(self, trainer, pl_module):
        print(f'Ended epcoh {trainer.current_epoch+epoch_offset}. Writing prediction for fragment id {self.fragment_id}')
        pl_module.eval()
        pred = predict_fn(self.loader, pl_module, self.device, self.xyxz, self.shape)
        pred = np.clip(np.nan_to_num(pred),a_min=0,a_max=1)
        pred/= pred.max()
        pred = (pred*255).astype(np.uint8)
        
        output = Image.fromarray(pred).resize((int(pred.shape[1]/4), int(pred.shape[0]/4)))
        save_name = f'{self.fragment_id}_epoch{trainer.current_epoch+epoch_offset}.png'
        output.save(f'{self.path}/{save_name}')

        # output prediction mask across different thresholds
        bg_mask = pred == 0
        for thresh in self.thresholds:
            output = np.copy(pred)
            ink_mask = pred > thresh*255
            output[ink_mask] = 255
            output[~ink_mask] = 60
            output[bg_mask] = 0
            output = Image.fromarray(output).resize((int(pred.shape[1]/4), int(pred.shape[0]/4)))
            output.save(f'{self.path}/thresholding/thresh_{thresh:.1}/{save_name}')

class BigCustomCallback(pl.Callback):
    """Save the prediction output for the grand prize fragment at end of predetermined epoch"""
    def __init__(self, dirpath, epochs=None):
        self.dirpath = dirpath
        self.thresholds = np.arange(0.3, 0.9, 0.1)
        self.epochs = [10, 15, 20, 25] if epochs is None else epochs
        self.fids = [
            '20230702185753', # shape=(13513, 17381)
            # '20231012173610', # shape=(31209, 10250)
            '20231012184422', # shape=(11882, 23364)
            # '20231022170900', # shape=(34588, 7676)
            '20231031143852', # shape=(10620, 13577)
            '20231106155351', # shape=(10636, 15738)
            '20231005123336', # shape=(10939, 29872)
            '20231012085431' # shape=(10187, 8565)
        ]
        for fid in self.fids:
            if not os.path.exists(f'{self.dirpath}/preds/grand_prize_{fid}'):
                os.makedirs(f'{self.dirpath}/preds/grand_prize_{fid}')
                for thresh in self.thresholds:
                    os.makedirs(f'{self.dirpath}/preds/grand_prize_{fid}/thresholding/thresh_{thresh:.1}')

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch+epoch_offset not in self.epochs:
            return
        pl_module.eval()
        for fid in self.fids:
            print(f'Ended epoch {trainer.current_epoch+epoch_offset}. Writing prediction for fragment id {fid}')
            pred = infer(fid, pl_module)

            output = Image.fromarray(pred).resize((int(pred.shape[1]/4), int(pred.shape[0]/4)))
            save_name = f'{fid}_epoch{trainer.current_epoch+epoch_offset}.png'
            output.save(f'{self.dirpath}/preds/grand_prize_{fid}/{save_name}')

            # output prediction mask across different thresholds
            bg_mask = pred == 0
            for thresh in self.thresholds:
                output = np.copy(pred)
                ink_mask = pred > thresh*255
                output[ink_mask] = 255
                output[~ink_mask] = 60
                output[bg_mask] = 0
                output.save(f'{self.dirpath}/preds/grand_prize_{fid}/thresholding/thresh_{thresh:.1}/{save_name}')

def main():
    # Prepare Dataset
    info = []
    for path in glob(f'{LABEL_PATH}/*.png'):
        filename = os.path.basename(path)
        if 'preview_img' in filename: continue
        assert filename.endswith(f'{iteration}.png')
        fragment_id, confidence, _ = filename.split('_')
        if not os.path.exists(f'{DATA_PATH}/{fragment_id}'):
            print(f'WARNING: Fragment id {fragment_id} does not exist in the data path. Skipping.')
            continue
        info.append([fragment_id, confidence, filename])
    df = pd.DataFrame(info, columns=['fragment_id', 'confidence', 'mask_filename'])
    data_paths, labels = get_data(df, write_fragment_chunks=False)

    # Create Model
    model = RegressionPLModel.load_from_checkpoint(MODEL_PATH, strict=False)

    # Create Trainer
    if not os.path.exists(f'./models/iteration{iteration+1}'):
        os.makedirs(f'./models/iteration{iteration+1}')
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        default_root_dir="./models",
        accumulate_grad_batches=8,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            ModelCheckpoint(
                filename=f'I3D_iteration{iteration+1}', # +'{epoch}',
                dirpath=f'./models/iteration{iteration+1}',
                monitor='train_loss_epoch',
                mode='min',
                save_top_k=-1 # save every model for now
            ),
            # SmallCustomCallback(
            #     # dirpath=f'./models/iteration{iteration+1}'
            #     dirpath=OUTPUT_PATH
            # ),
            # BigCustomCallback(
            #     # dirpath=f'./models/iteration{iteration+1}'
            #     dirpath=OUTPUT_PATH
            # )
        ]
    )

    # Setup data splits
    torch.set_float32_matmul_precision('medium')
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # train_idxs, valid_idxs = next(gss.split(labels, groups=data_paths))
    # train_paths, valid_paths, train_labels, valid_labels = data_paths[train_idxs], data_paths[valid_idxs], labels[train_idxs], labels[valid_idxs]
    # print(f'Train shape: {train_labels.shape}, Valid shape: {valid_paths.shape}')

    # Setup dataloader
    # train_dataset = CustomDataset(train_paths, labels=train_labels, transform=A.Compose(train_aug_list))
    # valid_dataset = CustomDataset(valid_paths, labels=valid_labels, transform=A.Compose(valid_aug_list))
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    dataset = CustomDataset(data_paths, labels=labels, transform=A.Compose(train_aug_list))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    # Train the model
    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)


if __name__ == '__main__':
    main()