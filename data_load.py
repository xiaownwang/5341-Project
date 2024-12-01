import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from ldm.data.open_images import OpenImageDataset, AugmentedOpenImageDataset

def process_dataset(dataset_name, coco_images_path, coco_annotations_file, output_images_path, output_annotations_file, num_images):

    os.makedirs(output_images_path, exist_ok=True)

    with open(coco_annotations_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"][:num_images]
    image_id_map = {img["id"]: img["file_name"] for img in images}

    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in image_id_map]

    for img in images:
        image_path = os.path.join(coco_images_path, img["file_name"])
        output_image_path = os.path.join(output_images_path, img["file_name"])

        output_image_path = os.path.splitext(output_image_path)[0] + ".jpeg"

        with Image.open(image_path) as image:
            if image.mode in ("RGBA", "P", "I"):
                image = image.convert("RGB")
            image.save(output_image_path, format="JPEG")

    output_coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": coco_data["categories"]
    }

    # Save the new annotations file
    with open(output_annotations_file, "w") as f:
        json.dump(output_coco_data, f, indent=4)

    print(
        f"{dataset_name} dataset processed. The first {num_images} images and annotations have been saved to the target folder.")

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
    num_images_to_process = 200  # Set the number of images to process

    for dataset in datasets:
        coco_images_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/{dataset}/val2017"
        coco_annotations_file = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/{dataset}/annotations/instances_val2017.json"
        output_images_path = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/{dataset}/images"
        output_annotations_file = f"/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/{dataset}/annotations.json"

        process_dataset(dataset, coco_images_path, coco_annotations_file, output_images_path, output_annotations_file,
                        num_images_to_process)

    # show samples after data preprocessing in open_images.py
    sample_show = OpenImageDataset(
        state='train',
        annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/annotations.json',
        coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/images',
        image_size=224
    )
    show_multiple_processed_images(sample_show, indices=[0, 1])

    sample_show = AugmentedOpenImageDataset(
        state='train',
        annotation_file='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/annotations.json',
        coco_root='/Users/xiaowenwang/PycharmProjects/Paint-by-Example-main/dataset/cartoon/images',
        image_size=224,
        style_aug_prob=0.7
    )
    show_multiple_processed_images(sample_show, indices=[0, 1])
