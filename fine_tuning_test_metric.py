import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import models
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
import clip
import torchvision.transforms as transforms
import warnings


def calculate_fid(real_images_path, generated_images_path, batch_size=50, device="cpu"):
    """
    Calculate Fréchet Inception Distance (FID)

    real_images_path: Path to reference images
    generated_images_path: Path to generated images
    batch_size: Number of images to process per batch
    device: Device to use ('cuda' or 'cpu')
    """
    # Load the InceptionV3 model
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    # Image preprocessing function
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception needs 299x299 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet standardization
    ])

    def get_features(image_paths):
        features = []
        with (torch.no_grad()):
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch = []
                for path in batch_paths:
                    img = Image.open(path)
                    img_resized = img.resize((299, 299))
                    img_resized= img_resized.convert("RGB")
                    img_resized = preprocess(img_resized).unsqueeze(0).to(device)
                    batch.append(img_resized)

                batch = torch.cat(batch)
                output = model(batch)
                features.append(output.cpu().numpy())
                # feature_vector = model.Conv2d_1a_3x3(batch)
                # features.append(feature_vector.cpu().numpy())

        return np.concatenate(features, axis=0)

    # Get the features of real and generated images
    real_images = [os.path.join(real_images_path, fname) for fname in os.listdir(real_images_path) if
                   fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    generated_images = [os.path.join(generated_images_path, fname) for fname in os.listdir(generated_images_path) if
                        fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    real_features = get_features(real_images)
    generated_features = get_features(generated_images)

    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)

    # Calculate FID
    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))

    # Handle possible imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return fid


def calculate_qs_score(real_images_path, generated_images_path, device="cpu"):
    """
    Calculate Quality Score (QS) based on Wasserstein distance between real and generated image distributions.

    :param real_images_path: Path to real images.
    :param generated_images_path: Path to generated images.
    :param device: Device to use ('cpu' or 'cuda').
    :return: QS score (lower is better).
    """
    # Load pretrained InceptionV3 model and modify for feature extraction
    model = models.inception_v3(pretrained=True).to(device)
    model.fc = torch.nn.Identity()  # Replace final layer with identity
    model.eval()

    # Preprocessing pipeline for images
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(image_paths, batch_size=32):
        """Extract features from a list of images."""
        features = []
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                for path in batch_paths:
                    img = Image.open(path).convert("RGB")
                    img = preprocess(img).unsqueeze(0).to(device)
                    batch_images.append(img)

                # Process batch
                batch_images = torch.cat(batch_images, dim=0)
                batch_features = model(batch_images)
                features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    # Load image paths
    real_images = [os.path.join(real_images_path, f) for f in os.listdir(real_images_path) if
                   f.lower().endswith(('png', 'jpg', 'jpeg'))]
    generated_images = [os.path.join(generated_images_path, f) for f in os.listdir(generated_images_path) if
                        f.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Extract features
    real_features = extract_features(real_images)
    generated_features = extract_features(generated_images)

    # Compute Wasserstein distances for each feature dimension
    qs_score = 0
    for i in range(real_features.shape[1]):
        w_dist = wasserstein_distance(real_features[:, i], generated_features[:, i])
        qs_score += w_dist

    return qs_score




def calculate_clip_score(real_images_path, generated_images_path, device="mps"):
    """
    Calculate CLIP score between real and generated images.

    real_images_path: Path to reference images
    generated_images_path: Path to generated images
    device: Device to use ('mps' or 'cpu')
    """
    # Load the CLIP model
    model, preprocess = clip.load("ViT-B/32", device)

    # Preprocess function for images
    def preprocess_image(image_path):
        image = Image.open(image_path).convert("RGB")
        return preprocess(image).unsqueeze(0).to(device)

    # Load and preprocess the images
    real_images = [os.path.join(real_images_path, fname) for fname in os.listdir(real_images_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    generated_images = [os.path.join(generated_images_path, fname) for fname in os.listdir(generated_images_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Get features for both real and generated images
    real_image_features = []
    generated_image_features = []

    for image_path in real_images:
        real_image = preprocess_image(image_path)
        real_image_features.append(model.encode_image(real_image))

    for image_path in generated_images:
        generated_image = preprocess_image(image_path)
        generated_image_features.append(model.encode_image(generated_image))

    # Normalize features
    real_image_features = torch.stack(real_image_features)
    generated_image_features = torch.stack(generated_image_features)

    real_image_features /= real_image_features.norm(dim=-1, keepdim=True)
    generated_image_features /= generated_image_features.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity between real and generated image features
    clip_score = 0
    for real_feature, generated_feature in zip(real_image_features, generated_image_features):
        similarity = torch.matmul(real_feature, generated_feature.T)
        clip_score += similarity.item()

    return clip_score / len(real_images)



if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)  # Ignore all UserWarnings

    parser = argparse.ArgumentParser(description="Calculate FID, QS, and CLIP Scores")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., 'cartoon')")
    parser.add_argument("--real_images_path", type=str, required=True, help="Path to real images")
    parser.add_argument("--reference_images_path", type=str, required=True, help="Path to reference images")
    parser.add_argument("--generated_images_path_ft", type=str, required=True, help="Path to generated images on fine-tuend model")
    parser.add_argument("--generated_images_path_p", type=str, required=True, help="Path to generated images on pretrained model")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    real_images_path = args.real_images_path
    reference_images_path = args.reference_images_path
    generated_images_path_ft = args.generated_images_path_ft
    generated_images_path_p = args.generated_images_path_p
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\n#########################################")
    print(f"Evaluation metric on {dataset_name} dataset")
    print(f"#########################################")

    print("On fine-tuend model:")
    # FID
    fid = calculate_fid(real_images_path, generated_images_path_ft, device=device)
    print(f"FID Score: {fid}")

    # QS (Quality Score)
    qs_score = calculate_qs_score(real_images_path, generated_images_path_ft, device=device)
    print(f"Quality Score (QS): {qs_score}")

    # CLIP score
    clip_score = calculate_clip_score(reference_images_path, generated_images_path_ft, device=device)
    print(f"CLIP Score: {clip_score}")

    print("\nOn original pretrained model:")
    # FID
    fid = calculate_fid(real_images_path, generated_images_path_p, device=device)
    print(f"FID Score: {fid}")

    # QS (Quality Score)
    qs_score = calculate_qs_score(real_images_path, generated_images_path_p, device=device)
    print(f"Quality Score (QS): {qs_score}")

    # CLIP score
    clip_score = calculate_clip_score(reference_images_path, generated_images_path_p, device=device)
    print(f"CLIP Score: {clip_score}")

