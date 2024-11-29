import os
import json
import random
import copy

import bezier
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import albumentations as A

class OpenImageDataset(data.Dataset):
    def __init__(self, state, arbitrary_mask_percent=0, annotation_file=None, coco_root=None, image_size=224, **args):
        """
        :param state: 'train', 'validation', or 'test'
        :param arbitrary_mask_percent: Masking percentage
        :param annotation_file: Path to the annotations JSON file
        :param coco_root: Path to the images directory
        :param image_size: Desired image size
        :param args: Additional arguments
        """
        self.state = state
        self.args = args
        self.arbitrary_mask_percent = arbitrary_mask_percent

        if annotation_file is None:
            raise ValueError("annotation_file must be provided")
        if coco_root is None:
            raise ValueError("coco_root must be provided")

        self.annotation_file = annotation_file
        self.coco_root = coco_root

        self.kernel = np.ones((1, 1), np.uint8)
        self.image_size = image_size

        # Load the annotation file
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        # Extract image info and annotations
        self.images = self.data['images']
        self.annotations = self.data['annotations']

        # Map image IDs to annotation data
        self.image_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)

        # self.image_ids = [img['id'] for img in self.images]

        # Split images into train, validation, and test sets
        all_image_ids = [img['id'] for img in self.images]
        train_ids, val_ids = train_test_split(all_image_ids, test_size=0.3, random_state=42)
        test_ids = []
        # train_ids, temp_ids = train_test_split(all_image_ids, test_size=0.3, random_state=42)
        # test_ids, val_ids = train_test_split(temp_ids, test_size=1, random_state=42)

        if self.state == 'train':
            self.image_ids = train_ids
        elif self.state == 'validation':
            self.image_ids = val_ids
        elif self.state == 'test':
            self.image_ids =  test_ids

        self.length = len(self.image_ids)

        # Define transformations
        self.random_trans = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20),
            A.Blur(p=0.3),
            A.ElasticTransform(p=0.3)
        ])

    def __getitem__(self, index):
        img_id = self.image_ids[index]

        # Get image info
        img_info = next(item for item in self.images if item['id'] == img_id)
        img_path = os.path.join(self.coco_root, img_info['file_name'])

        # Load image
        img_p = Image.open(img_path).convert("RGB")
        W, H = img_p.size

        # Get annotations for this image
        annotations = self.image_to_annotations.get(img_id, [])

        # Randomly choose one bbox
        bbox = random.choice(annotations)['bbox']

        # Process bbox to get (x_min, y_min, x_max, y_max)
        bbox = self.bbox_process(bbox)

        # Get reference image
        ref_image_tensor = self.get_reference_image(img_p, bbox)
        image_tensor = self.get_tensor()(img_p)  # Resize and return

        # Generate mask
        mask_tensor = self.generate_mask(img_p, bbox, W, H)

        # Crop image and mask using bbox
        image_tensor_cropped, mask_tensor_cropped = self.crop_image(image_tensor, mask_tensor, bbox, img_p.size)

        image_tensor_resize = T.Resize([self.image_size, self.image_size])(image_tensor_cropped)
        mask_tensor_resize = T.Resize([self.image_size, self.image_size])(mask_tensor_cropped)
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize

        return {
            "GT": image_tensor_resize,
            "inpaint_image": inpaint_tensor_resize,
            "inpaint_mask": mask_tensor_resize,
            "ref_imgs": ref_image_tensor
        }

    def __len__(self):
        return self.length

    def get_tensor(self):
        """
        Returns a tensor transformation with normalization.
        """
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_tensor_clip(self):
        """
        Returns a tensor transformation without normalization.
        """
        return T.Compose([
            T.ToTensor(),
        ])

    def bbox_process(self, bbox):
        """
        Processes the bounding box into (x_min, y_min, x_max, y_max).
        """
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return [x_min, y_min, x_max, y_max]

    def get_reference_image(self, img_p, bbox):
        """
        Crop and process the reference image from the original image based on the bbox.
        """
        bbox_pad = copy.copy(bbox)
        bbox_pad[0] = bbox[0] - min(10, bbox[0] - 0)
        bbox_pad[1] = bbox[1] - min(10, bbox[1] - 0)
        bbox_pad[2] = bbox[2] + min(10, img_p.size[0] - bbox[2])
        bbox_pad[3] = bbox[3] + min(10, img_p.size[1] - bbox[3])

        ref_image_tensor = img_p.crop((bbox_pad[0], bbox_pad[1], bbox_pad[2], bbox_pad[3]))
        ref_image_tensor = self.random_trans(image=np.array(ref_image_tensor))['image']
        ref_image_tensor = Image.fromarray(ref_image_tensor)
        ref_image_tensor = self.get_tensor_clip()(ref_image_tensor)

        return ref_image_tensor

    def generate_mask(self, img_p, bbox, W, H):
        """
        Generate a mask for the image based on the bbox.
        """
        extended_bbox = self.extend_bbox(bbox, W, H)
        prob = random.uniform(0, 1)

        if prob < self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255))
            mask_img = self.create_arbitrary_mask(mask_img, bbox, extended_bbox)
            mask_img = mask_img.convert("RGB")
            mask_tensor = self.get_tensor()(mask_img)[0].unsqueeze(0)
            mask_tensor = mask_tensor.repeat(3, 1, 1)
        else:
            mask_img = np.zeros((H, W))
            mask_img[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]] = 1
            mask_img = Image.fromarray(mask_img)
            mask_img = mask_img.convert("RGB")
            mask_tensor = 1 - self.get_tensor()(mask_img)

        return mask_tensor

    def extend_bbox(self, bbox, W, H):
        """
        Extend the bbox to include some surrounding area.
        """
        extended_bbox = copy.copy(bbox)

        left_freespace = max(0, bbox[0])
        right_freespace = max(0, W - bbox[2])
        up_freespace = max(0, bbox[1])
        down_freespace = max(0, H - bbox[3])

        max_left_offset = max(0, int(0.4 * left_freespace))
        max_up_offset = max(0, int(0.4 * up_freespace))
        max_right_offset = max(0, int(0.4 * right_freespace))
        max_down_offset = max(0, int(0.4 * down_freespace))

        extended_bbox[0] = int(bbox[0] - random.randint(0, max_left_offset))
        extended_bbox[1] = int(bbox[1] - random.randint(0, max_up_offset))
        extended_bbox[2] = int(bbox[2] + random.randint(0, max_right_offset))
        extended_bbox[3] = int(bbox[3] + random.randint(0, max_down_offset))

        return extended_bbox

    def create_arbitrary_mask(self, mask_img, bbox, extended_bbox):
        """
        Create an arbitrary mask using bezier curves or rectangle shapes for a given bbox.
        """
        W, H = mask_img.size
        mask_img = Image.new('RGB', (W, H), (255, 255, 255))  # Create a blank white mask

        prob = random.uniform(0, 1)
        if prob < self.arbitrary_mask_percent:  # Use bezier curve for mask
            bbox_mask = copy.copy(bbox)
            extended_bbox_mask = copy.copy(extended_bbox)

            # Create Bezier curves for the four sides
            top_nodes = np.asfortranarray([
                [bbox_mask[0], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[2]],
                [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
            ])
            down_nodes = np.asfortranarray([
                [bbox_mask[2], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[0]],
                [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
            ])
            left_nodes = np.asfortranarray([
                [bbox_mask[0], extended_bbox_mask[0], bbox_mask[0]],
                [bbox_mask[3], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[1]],
            ])
            right_nodes = np.asfortranarray([
                [bbox_mask[2], extended_bbox_mask[2], bbox_mask[2]],
                [bbox_mask[1], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[3]],
            ])

            # Create curves
            top_curve = bezier.Curve(top_nodes, degree=2)
            right_curve = bezier.Curve(right_nodes, degree=2)
            down_curve = bezier.Curve(down_nodes, degree=2)
            left_curve = bezier.Curve(left_nodes, degree=2)

            # Collect points from curves
            curve_list = [top_curve, right_curve, down_curve, left_curve]
            pt_list = []
            random_width = 5
            for curve in curve_list:
                x_list = []
                y_list = []
                for i in range(1, 19):
                    point = curve.evaluate(i * 0.05)
                    if point[0][0] not in x_list and point[1][0] not in y_list:
                        pt_list.append((point[0][0] + random.randint(-random_width, random_width),
                                        point[1][0] + random.randint(-random_width, random_width)))
                        x_list.append(point[0][0])
                        y_list.append(point[1][0])

            # Draw the polygon based on points
            mask_img_draw = ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list, fill=(0, 0, 0))  # Fill with black

        else:  # Use simple rectangular mask
            mask_img_np = np.zeros((H, W))
            mask_img_np[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]] = 1
            mask_img = Image.fromarray(mask_img_np)

        return mask_img

    def crop_image(self, image_tensor, mask_tensor, bbox, img_size):
        """
        Crop the image and the mask based on the bounding box and image size.
        The crop will be based on the relative width and height of the image.
        """
        W, H = img_size
        extended_bbox = self.extend_bbox(bbox, W, H)  # Ensure bbox is extended if needed

        # Initialize the cropped tensors to the full image and mask by default
        image_tensor_cropped = image_tensor
        mask_tensor_cropped = mask_tensor

        try:
            # Handle case where width is greater than height
            if W > H:
                left_most = max(0, extended_bbox[2] - H)
                right_most = min(W, extended_bbox[0] + H) - H

                if right_most > left_most:
                    left_pos = random.randint(left_most, right_most)
                    free_space = max(0, min(
                        extended_bbox[1], extended_bbox[0] - left_pos,
                                          left_pos + H - extended_bbox[2], H - extended_bbox[3]
                    ))
                    random_free_space = random.randint(0, max(1, int(0.6 * free_space)))
                    image_tensor_cropped = image_tensor[:, random_free_space:H - random_free_space,
                                           left_pos + random_free_space:left_pos + H - random_free_space]
                    mask_tensor_cropped = mask_tensor[:, random_free_space:H - random_free_space,
                                          left_pos + random_free_space:left_pos + H - random_free_space]

            # Handle case where width is less than height
            elif W < H:
                upper_most = max(0, extended_bbox[3] - W)
                lower_most = min(H, extended_bbox[1] + W) - W

                if lower_most > upper_most:
                    upper_pos = random.randint(upper_most, lower_most)
                    free_space = max(0, min(
                        extended_bbox[1] - upper_pos, extended_bbox[0],
                        W - extended_bbox[2], upper_pos + W - extended_bbox[3]
                    ))
                    random_free_space = random.randint(0, max(1, int(0.6 * free_space)))
                    image_tensor_cropped = image_tensor[:,
                                           upper_pos + random_free_space:upper_pos + W - random_free_space,
                                           random_free_space:W - random_free_space]
                    mask_tensor_cropped = mask_tensor[:,
                                          upper_pos + random_free_space:upper_pos + W - random_free_space,
                                          random_free_space:W - random_free_space]
        except Exception as e:
            print(f"Crop error: {e}, bbox: {bbox}, extended_bbox: {extended_bbox}, image size: {img_size}")

        return image_tensor_cropped, mask_tensor_cropped

# # Example usage
# dataset = OpenImageDataset(state='validation',
#                              annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/COCO_O/annotations/instances_val2017.json',
#                              coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/COCO_O/images',
#                              arbitrary_mask_percent=0.5,
#                              image_size=224)
#
# print(len(dataset))
# sample = dataset[i]
# # print(sample.keys())
#
# print("GT shape:", sample['GT'].shape)
# print("inpaint_image shape:", sample['inpaint_image'].shape)
# print("inpaint_mask shape:", sample['inpaint_mask'].shape)
# print("ref_imgs shape:", sample['ref_imgs'].shape)


class AugmentedOpenImageDataset(OpenImageDataset):
    def __init__(self, *args, style_aug_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Extends the OpenImageDataset class with additional style-specific augmentations to simulate artistic variations.

        :param style_aug_prob: Probability of applying artistic style augmentations to a given image
        :param args: Positional arguments passed to the base OpenImageDataset class
        :param kwargs: Keyword arguments passed to the base OpenImageDataset class
        """
        # Artistic augmentations
        self.style_aug_prob = style_aug_prob
        self.style_augmentations = A.Compose([
            A.OneOf([
                # Color Adjustments: Modifies brightness, contrast, saturation, and hue
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            ]),
            A.OneOf([
                # Blur Effects: Add artistic blur effects
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ]),
            A.OneOf([
                # Geometric Distortions: Warp the image to simulate hand-drawn or abstract effects
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
            ]),
            A.ToGray(p=0.2),  # Convert to grayscale with a small probability
        ])

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset and applies optional style augmentations.
        """
        sample = super().__getitem__(index)

        if random.random() < self.style_aug_prob:
            # Apply artistic augmentations
            sample["GT"] = self.apply_style_augmentations(sample["GT"])
            sample["inpaint_image"] = self.apply_style_augmentations(sample["inpaint_image"])
        
        return sample

    def apply_style_augmentations(self, image_tensor):
        """
        Apply style-specific augmentations to an image tensor.
        """
        image_np = image_tensor.permute(1, 2, 0).numpy() * 255  # Convert tensor to numpy
        image_np = image_np.astype(np.uint8)
        
        augmented = self.style_augmentations(image=image_np)
        image_aug = augmented["image"]

        image_tensor_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float() / 255.0
        return image_tensor_aug

# # Example usage
# augmented_dataset = AugmentedOpenImageDataset(
#     state='train',
#     annotation_file='path/to/annotations.json',
#     coco_root='path/to/images',
#     arbitrary_mask_percent=0.5,
#     image_size=224,
#     style_aug_prob=0.7  # prob of applying style augmentations
# )

# print("Dataset length:", len(augmented_dataset))

# # fetch a sample
# sample_index = 0  # ex index
# sample = augmented_dataset[sample_index]

# print("GT shape:", sample['GT'].shape)              
# print("inpaint_image shape:", sample['inpaint_image'].shape)
# print("inpaint_mask shape:", sample['inpaint_mask'].shape)
# print("ref_imgs shape:", sample['ref_imgs'].shape)

# # test artistic augmentations (applied probabilistically)
# if 'GT_augmented' in sample:
#     print("Augmented ground truth shape:", sample['GT_augmented'].shape)

