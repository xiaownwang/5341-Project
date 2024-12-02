import os
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from ldm.data.open_images import OpenImageDataset, AugmentedOpenImageDataset


def load_train_data(dataset_name, coco_images_path, coco_annotations_file,
                    train_images_path, train_annotations_file, num_train_images):

    os.makedirs(train_images_path, exist_ok=True)

    with open(coco_annotations_file, "r") as f:
        coco_data = json.load(f)

    ## Training Dataset
    train_images = coco_data["images"][:num_train_images]
    train_ann_id = {img["id"]: img["file_name"] for img in train_images}
    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_ann_id]

    for img in train_images:
        image_path = os.path.join(coco_images_path, img["file_name"])
        output_image_path = os.path.join(train_images_path, img["file_name"])

        output_image_path = os.path.splitext(output_image_path)[0] + ".jpeg"

        with Image.open(image_path) as image:
            if image.mode in ("RGBA", "P", "I"):
                image = image.convert("RGB")
            image.save(output_image_path, format="JPEG")

    output_coco_data_train = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"]
    }

    # Save the new annotations file for train
    with open(train_annotations_file, "w") as f:
        json.dump(output_coco_data_train, f, indent=4)

    print(f"{dataset_name} training data processed. The first {num_train_images} images saved to {train_images_path} and annotations saved to {train_annotations_file}.")


def load_test_data(dataset_name, coco_annotations_file, test_image_selected, test_reference_selected, test_image_path, test_mask_path, test_reference_path):

    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(test_mask_path, exist_ok=True)
    os.makedirs(test_reference_path, exist_ok=True)

    coco = COCO(coco_annotations_file)

    image_files = os.listdir(test_image_selected)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.png', '.jpeg'))]

    image_ids = []
    for image_file in image_files:
        image_id_str = image_file.lstrip('0').rstrip('.jpeg')
        if image_id_str.isdigit():
            image_ids.append(int(image_id_str))

    for idx, image_id in enumerate(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(test_image_selected, image_info["file_name"])

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image_resized = cv2.resize(image, (512, 512))

        annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            mask[y:y + h, x:x + w] = 1

        mask = mask * 255
        mask_resized = cv2.resize(mask, (512, 512))

        img_filename = f"img_{idx + 1}.png"
        mask_filename = f"mask_{idx + 1}.png"

        cv2.imwrite(os.path.join(test_image_path, img_filename), image_resized)
        cv2.imwrite(os.path.join(test_mask_path, mask_filename), mask_resized)


    reference_files = os.listdir(test_reference_selected)
    reference_files = [f for f in reference_files if f.endswith(('.jpg', '.png', '.jpeg'))]

    for idx, reference_file in enumerate(reference_files):
        reference_image_path = os.path.join(test_reference_selected, reference_file)

        reference_image = cv2.imread(reference_image_path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        reference_image_resized = cv2.resize(reference_image, (512, 512))

        ref_filename = f"ref_{idx + 1}.png"
        cv2.imwrite(os.path.join(test_reference_path, ref_filename),
                    cv2.cvtColor(reference_image_resized, cv2.COLOR_RGB2BGR))

    print(f"{dataset_name} testing data processed. The results saved to dataset/testdata/{dataset_name}\n")


def show_multiple_processed_images(dataset, indices=None, max_samples=4):
    """
    Visualize the processed images and masks for multiple samples in the dataset.

    :param dataset: The dataset object.
    :param indices: List of indices to display. If None, display the first `max_samples` images.
    :param max_samples: Maximum number of samples to display if indices is not provided.
    """
    if indices is None:
        indices = list(range(min(len(dataset), max_samples)))

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))

    for i, idx in enumerate(indices):
        data = dataset[idx]

        def tensor_to_image(tensor):
            tensor = tensor.clone()  # Avoid in-place modification
            tensor = T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            )(tensor)
            tensor = tensor.permute(1, 2, 0).clamp(0, 1)  # C x H x W -> H x W x C
            return Image.fromarray((tensor.numpy() * 255).astype('uint8'))

        # Extract images and masks
        gt_image = tensor_to_image(data["GT"])
        inpaint_image = tensor_to_image(data["inpaint_image"])
        inpaint_mask = Image.fromarray((data["inpaint_mask"][0].numpy() * 255).astype('uint8'))
        ref_image = T.ToPILImage()(data["ref_imgs"])  # Convert reference image

        # Plot images
        axes[i][0].imshow(gt_image)
        axes[i][0].set_title(f"Sample {idx} - Ground Truth")
        axes[i][1].imshow(inpaint_image)
        axes[i][1].set_title(f"Sample {idx} - Inpainted Image")
        axes[i][2].imshow(inpaint_mask, cmap="gray")
        axes[i][2].set_title(f"Sample {idx} - Inpaint Mask")
        axes[i][3].imshow(ref_image)
        axes[i][3].set_title(f"Sample {idx} - Reference Image")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    datasets = ["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"]
    num_train_images = 200

    for dataset in datasets:
        coco_images_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/ood_coco/{dataset}/val2017"
        coco_annotations_file = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/ood_coco/{dataset}/annotations/instances_val2017.json"
        train_images_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/{dataset}/images"
        train_annotations_file = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/{dataset}/annotations.json"

        test_image_selected = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/testdata_selected/{dataset}/image"
        test_reference_selected = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/testdata_selected/{dataset}/reference"

        test_image_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/testdata/{dataset}/image"
        test_mask_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/testdata/{dataset}/mask"
        test_reference_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/testdata/{dataset}/reference"

        load_train_data(dataset, coco_images_path, coco_annotations_file,
                        train_images_path, train_annotations_file, num_train_images)
        load_test_data(dataset, coco_annotations_file, test_image_selected, test_reference_selected, test_image_path, test_mask_path, test_reference_path)

    # show samples after data preprocessing in open_images.py
    sample_show = OpenImageDataset(
        state='train',
        annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/annotations.json',
        coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/images',
        image_size=224
    )
    show_multiple_processed_images(sample_show, indices=[0, 1])

    sample_show = AugmentedOpenImageDataset(
        state='train',
        annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/annotations.json',
        coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/traindata/cartoon/images',
        image_size=224,
        style_aug_prob=0.7
    )
    show_multiple_processed_images(sample_show, indices=[0, 1])

