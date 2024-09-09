import guided_diffusion.mapping_utils as mapping_utils
import numpy as np
from skimage.transform import resize
import torchvision.transforms as transforms
import random
import torch
from torch.utils.data import Dataset
from collections import namedtuple

TEST_PATIENTS = [7, 21, 30, 33, 34, 37, 41, 58, 86, 110, 123, 135, 145, 148, 155, 163, 164, 172, 177, 183, 190, 191, 207, 212, 220]
VAL_PATIENTS  = []#[3, 4, 12, 14, 19, 23, 28, 35, 40, 46, 50, 55, 98, 107, 130, 137, 156, 162, 176, 182, 185, 197, 209, 213, 219]
TRAIN_PATIENTS = [id for id in range(1, 222+1) if id not in TEST_PATIENTS and id not in VAL_PATIENTS]
#TRAIN_PATIENTS = [1, 2, 3]
INTEROBSERVER_PATIENTS = range(223, 262+1)

# normalize?? -1, 1 közé
class SeDataset(Dataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):        
        parser.add_argument('--split', type=str, default="train", help="train, val, test, interobserver")
        parser.add_argument('--observer_id', type=int, default=1, help="1, 2, 3")
        parser.add_argument('--mapping_only', action='store_true', help="Only use mapping samples")
        
        
        parser.set_defaults(input_nc=1)
        parser.set_defaults(output_nc=1)
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(no_flip=True)

        return parser

    def __init__(self,
            data_root,
            resolution,
            classes=3,
            instances=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=False,
            is_train=True):
        super().__init__()


        self.transforms = None#transforms
        self.split = 'train' if is_train else 'test'
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.resolution = resolution
        '''
        if self.split == "interobserver" and opt.observer_id != 1:
            contours_filename = f"Contours_{opt.observer_id}.json"
        else:
            contours_filename = f"Contours.json"
        '''
        contours_filename = f"Contours.json"

        self.all_samples = mapping_utils.construct_samples_list(
            data_root, contours_filename
        )
        mapping_utils.print_diagnostics(data_root, self.all_samples)

        segmentation_partitions = {
            "train": TRAIN_PATIENTS,
            "val": VAL_PATIENTS,
            "test": TEST_PATIENTS,
            "interobserver": INTEROBSERVER_PATIENTS,
        }


        self.samples = mapping_utils.split_samples_list(
            self.all_samples, segmentation_partitions[self.split]
        )
        #if opt.mapping_only:
            #self.to_mapping_only()

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict: (sample, mask)
        """
        path, mask_path = self.samples[index]
        sample = mapping_utils.load_dicom(path, mode=None)
        mask_contours = mapping_utils.load_contours(mask_path)
        mask = mapping_utils.contours_to_masks(mask_contours, sample.shape)
        if self.transforms is not None:
            if "bboxes" in self.transforms.processors.keys():
                bbox = self.compute_bbox(mask)
                transformed = self.transforms(image=sample, mask=mask, bboxes=[bbox])
            else:
                transformed = self.transforms(image=sample, mask=mask)
            sample, mask = transformed["image"], transformed["mask"]
        
        # Convert images to channels_first mode, from albumentations' 2d grayscale images
        sample = resize(sample, (self.resolution, self.resolution), anti_aliasing=True)
        
        mask = resize(mask, (self.resolution, self.resolution), anti_aliasing=False, mode='edge', preserve_range=True, order=0)

        sample = np.expand_dims(sample, 0)

        # repeate the first channel 3 times to make it 3 channel
        # sample = np.repeat(sample, 3, axis=0)

        mask = np.expand_dims(mask, 0)

        sample = sample.astype(np.float32)
        mask = mask.astype(np.float32)
        
        if self.random_crop and random.random() < 0.33:
            opt = {
                'preprocess': 'resize_and_crop',      
                'crop_size': int(self.resolution),
                'load_size': int(self.resolution * 1.5),
                'flip': self.random_flip,
            }
            
            transform_params = get_params(opt, sample.shape[1:])
            B_transform = get_transform(opt, transform_params, method=transforms.InterpolationMode.NEAREST, grayscale=True, convert=False)
            A_transform = get_transform(opt, transform_params, grayscale=True, convert=False)
                      
            sample = torch.from_numpy(sample)
            mask = torch.from_numpy(mask)
            
            sample = A_transform(sample)
            mask = B_transform(mask)

            sample = sample.numpy()
            mask = mask.numpy()
        
        mask = np.squeeze(mask, axis=0)
        sample = (sample - np.min(sample.flatten())) / (np.max(sample.flatten()) - np.min(sample.flatten()))
        sample = (sample * 2) - 1
        
        out_dict = {}
        out_dict['path'] = path
        out_dict['label_ori'] = mask.copy()
        out_dict['label'] = mask#mask[None,]


        return sample, out_dict

    def compute_bbox(self, mask):
        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        bbox = [
            mask.nonzero()[1].min(),  # / mask.shape[0],
            mask.nonzero()[0].min(),  # / mask.shape[1],
            mask.nonzero()[1].max(),  # / mask.shape[0],
            mask.nonzero()[0].max(),  # / mask.shape[1],
            "dummy_label",
        ]  # x_min, y_min, x_max, y_max
        return bbox

    def __len__(self) -> int:
        return len(self.samples)

    def to_mapping_only(self):
        self.samples = [(x, t) for x, t in self.samples if "_Mapping_" in x]


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['load_size']
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w

    x = random.randint(0, np.maximum(0, new_w - opt['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_size']))


    return {'crop_pos': (x, y)}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    #if grayscale:
    #    transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt['preprocess']:
        osize = [opt['load_size'], opt['load_size']]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt['preprocess']:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt['load_size'], opt['crop_size'], method)))

    if 'crop' in opt['preprocess']:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['crop_size'])))

    if opt['preprocess'] == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if opt['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        #transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __crop(img, pos, size):
    ow, oh = img.shape[1:]
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img[:, y1:y1 + th, x1:x1 + tw]
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
