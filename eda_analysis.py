import os
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    dataset_path = os.path.join("data", "extracted", "Data")
    print(f"Scanning dataset at: {dataset_path}...")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        return

    class_counts = {}
    corrupted_images = []
    image_shapes = set()

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    for class_name in classes:
        class_folder = os.path.join(dataset_path, class_name)
        image_files = glob.glob(os.path.join(class_folder, "*.*"))
        class_counts[class_name] = len(image_files)

        # Quick integrity and shape check on sample images
        for img_path in image_files[:5]:
            img = cv2.imread(img_path)
            if img is None:
                corrupted_images.append(img_path)
            else:
                image_shapes.add(img.shape)

    df = pd.DataFrame(list(class_counts.items()), columns=["Sign Class", "Image Count"])

    print("\n" + "="*45)
    print(f"Total Sign Classes     : {len(df)}")
    print(f"Total Images           : {df['Image Count'].sum()}")
    print(f"Sample Image Shapes    : {list(image_shapes)}")
    print(f"Corrupted Images Count : {len(corrupted_images)}")
    print("="*45)
    print(df.to_string(index=False))

    # Generate and save distribution plot
    plt.figure(figsize=(12, 5))
    sns.barplot(x="Sign Class", y="Image Count", data=df, hue="Sign Class", palette="viridis", legend=False)
    plt.title("Indian Sign Language (ISL) Dataset - Class Distribution", fontsize=14, weight="bold")
    plt.xlabel("Sign Letter Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.axhline(df["Image Count"].mean(), color="red", linestyle="--", label=f"Mean Count ({df['Image Count'].mean():.1f})")
    plt.legend()
    plt.tight_layout()

    output_plot = "isl_dataset_distribution.png"
    plt.savefig(output_plot, dpi=300)
    print(f"\n[SUCCESS] Class distribution plot saved as: '{output_plot}'")

if __name__ == "__main__":
    main()

