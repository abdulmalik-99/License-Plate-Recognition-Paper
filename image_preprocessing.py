import os
import cv2
import random
import yaml
import shutil
import os
from torchvision import transforms
import argparse
import cv2
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess and split data for YOLO model training.")
    parser.add_argument("--images_dir", required=True, help="Directory path for the input images.")
    parser.add_argument("--annotations_dir", required=True, help="Directory path for the input annotations.")
    parser.add_argument("--output_dir", required=True, help="Directory path for the output preprocessed and split data.")
    parser.add_argument("--names", required=True, type=str, help="Comma-separated list of class names.")
    parser.add_argument("--nc", type=int, required=True, help="Number of classes.")
    parser.add_argument("--split_ratio", nargs=3, type=float, default=[0.7, 0.2, 0.1], help="Split ratio for training, validation, and testing sets.")
    return parser.parse_args()
def image_preprocessing(image):
    #  Resize image to 640x640
    resized_image = cv2.resize(image, (640, 640))

    #  Convert image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    #  Remove noise from the image
    denoised_image = cv2.fastNlMeansDenoising(gray_image)

    # Increase brightness of the image
    brightness_transform = transforms.ColorJitter(brightness=0.2)
    brightened_image = brightness_transform(denoised_image)

    _, binary_image = cv2.adaptiveThreshold(brightened_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)

    return binary_image

def preprocess_image(image_path, annotation_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Perform image pre-processing
    preprocessed_image = image_preprocessing(image)

    # Generate the output path for the preprocessed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)

    # Save the preprocessed image
    cv2.imwrite(output_path, preprocessed_image)

    # Adjust the annotation coordinates
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    with open(os.path.join(output_dir, os.path.basename(annotation_path)), 'w') as file:
        for line in lines:
            values = line.strip().split()
            class_label = values[0]
            x = float(values[1]) * 640 / image.shape[1]
            y = float(values[2]) * 640 / image.shape[0]
            width = float(values[3]) * 640 / image.shape[1]
            height = float(values[4]) * 640 / image.shape[0]
            file.write(f"{class_label} {x} {y} {width} {height}\n")

    return output_path

def split_data(images_dir, annotations_dir, output_dir, split_ratio):
    # Create output directories for the split sets
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get a list of all image files in the directory
    image_files = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]

    # Shuffle the image files randomly
    random.shuffle(image_files)

    # Calculate the number of images for each split set
    num_images = len(image_files)
    num_train = int(num_images * split_ratio[0])
    num_valid = int(num_images * split_ratio[1])
    num_test = num_images - num_train - num_valid

    train_files = image_files[:num_train]
    valid_files = image_files[num_train:num_train+num_valid]
    test_files = image_files[num_train+num_valid:]


    for file in train_files:
            image_path = os.path.join(images_dir, file)
            annotation_path = os.path.join(annotations_dir, file.replace('.jpg', '.txt'))
            output_path = preprocess_image(image_path, annotation_path, train_dir)
            # Move the corresponding annotation file
            shutil.move(annotation_path, os.path.join(train_dir, os.path.basename(annotation_path)))

    for file in valid_files:
        image_path = os.path.join(images_dir, file)
        annotation_path = os.path.join(annotations_dir, file.replace('.jpg', '.txt'))
        output_path = preprocess_image(image_path, annotation_path, valid_dir)
        # Move the corresponding annotation file
        shutil.move(annotation_path, os.path.join(valid_dir, os.path.basename(annotation_path)))

    for file in test_files:
        image_path = os.path.join(images_dir, file)
        annotation_path = os.path.join(annotations_dir, file.replace('.jpg', '.txt'))
        output_path = preprocess_image(image_path, annotation_path, test_dir)
        # Move the corresponding annotation file
        shutil.move(annotation_path, os.path.join(test_dir, os.path.basename(annotation_path)))

def create_yaml_file(output_dir,nc,class_names):
    yaml_data = {
        'train': 'train/',
        'val': 'valid/',
        'test': 'test/',
        'nc': nc,
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')

    with open(yaml_path, 'w') as file:
        documents = yaml.dump(yaml_data, file)

    print(f"Created YAML file: {yaml_path}")



if __name__ == "__main__":
    args = parse_arguments()

    # Split the data into sets
    split_data(args.images_dir, args.annotations_dir, args.output_dir, args.split_ratio)

    # Create the YAML file
    class_names = args.names.split(',')
    create_yaml_file(args.output_dir, args.nc, class_names)