import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # For progress bar

def analyze_scene_changes(t0_folder, t1_folder, mask_folder, output_csv="vl_cmu_scene_changes.csv"):
    """
    Analyze potential scene changes using t0, t1, and mask folders.
    Outputs a CSV file with scene change information.
    
    Parameters:
    t0_folder: Path to folder containing frames before potential scene changes
    t1_folder: Path to folder containing frames after potential scene changes
    mask_folder: Path to folder containing masks showing what changed
    output_csv: Filename for the output CSV
    """
    # Get sorted lists of files
    t0_folder="VL-CMU-CD-binary255/train/t0"
    t1_folder = "VL-CMU-CD-binary255/train/t1"
    mask_folder = "VL-CMU-CD-binary255/train/mask"
    t0_files = sorted([f for f in os.listdir(t0_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    t1_files = sorted([f for f in os.listdir(t1_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Ensure we have matching files
    if not (len(t0_files) == len(t1_files) == len(mask_files)):
        raise ValueError(f"Mismatched number of files: t0={len(t0_files)}, t1={len(t1_files)}, mask={len(mask_files)}")
    
    # Prepare results data
    results = []
    
    # Process each frame set
    print(f"Processing {len(t0_files)} frame sets...")
    for i, (t0_file, t1_file, mask_file) in enumerate(tqdm(zip(t0_files, t1_files, mask_files), total=len(t0_files))):
        # Extract frame ID (assuming filenames contain identifiers)
        frame_id = os.path.splitext(t0_file)[0]
        
        # Load images
        img_t0 = cv2.imread(os.path.join(t0_folder, t0_file))
        img_t1 = cv2.imread(os.path.join(t1_folder, t1_file))
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        
        # Calculate mask statistics
        mask_pixel_count = np.sum(mask > 0)  # Count non-zero pixels in mask
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_percentage = (mask_pixel_count / total_pixels) * 100
        
        # Calculate difference between t0 and t1 (only in masked region)
        if mask_pixel_count > 0:  # Only if mask has non-zero pixels
            # Convert to grayscale for simpler comparison
            gray_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
            gray_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
            
            # Apply mask
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            masked_t0 = cv2.bitwise_and(gray_t0, gray_t0, mask=mask_binary)
            masked_t1 = cv2.bitwise_and(gray_t1, gray_t1, mask=mask_binary)
            
            # Calculate difference statistics within masked region
            diff = cv2.absdiff(masked_t0, masked_t1)
            mean_diff = np.sum(diff) / mask_pixel_count if mask_pixel_count > 0 else 0
            max_diff = np.max(diff)
        else:
            mean_diff = 0
            max_diff = 0
        
        # Determine if a scene change occurred (thresholds can be adjusted)
        # Here we're using both the mask size and the difference within the mask
        scene_change = (mask_percentage > 1.0) and (mean_diff > 10.0)
        
        # Store results
        results.append({
            'frame_id': frame_id,
            't0_file': t0_file,
            't1_file': t1_file,
            'mask_file': mask_file,
            'mask_percentage': mask_percentage,
            'mean_difference': mean_diff,
            'max_difference': max_diff,
            'scene_change': scene_change
        })
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Print summary
    scene_changes = df['scene_change'].sum()
    print(f"\nDetected {scene_changes} scene changes out of {len(df)} frames ({scene_changes/len(df)*100:.1f}%)")
    
    return df

def visualize_examples(df, t0_folder, t1_folder, mask_folder, output_folder="scene_examples"):
    """
    Save visual examples of scene changes and non-changes for review
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Get some examples of scene changes and non-changes
    scene_changes = df[df['scene_change'] == True].head(5)
    non_changes = df[df['scene_change'] == False].head(5)
    
    # Function to create comparison image
    def create_comparison(row):
        t0_img = cv2.imread(os.path.join(t0_folder, row['t0_file']))
        t1_img = cv2.imread(os.path.join(t1_folder, row['t1_file']))
        mask_img = cv2.imread(os.path.join(mask_folder, row['mask_file']))
        
        # Resize if images are different sizes
        height = min(t0_img.shape[0], t1_img.shape[0], mask_img.shape[0])
        width = min(t0_img.shape[1], t1_img.shape[1], mask_img.shape[1])
        
        t0_img = cv2.resize(t0_img, (width, height))
        t1_img = cv2.resize(t1_img, (width, height))
        mask_img = cv2.resize(mask_img, (width, height))
        
        # Create side-by-side comparison
        comparison = np.hstack((t0_img, t1_img, mask_img))
        
        # Add text labels
        cv2.putText(comparison, "t0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "t1", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "mask", (2*width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add metadata
        info_text = f"Frame: {row['frame_id']} | Mask: {row['mask_percentage']:.2f}% | Diff: {row['mean_difference']:.2f}"
        cv2.putText(comparison, info_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return comparison
    
    # Save examples
    for i, row in scene_changes.iterrows():
        comparison = create_comparison(row)
        cv2.imwrite(os.path.join(output_folder, f"scene_change_{row['frame_id']}.jpg"), comparison)
    
    for i, row in non_changes.iterrows():
        comparison = create_comparison(row)
        cv2.imwrite(os.path.join(output_folder, f"no_change_{row['frame_id']}.jpg"), comparison)
    
    print(f"Example visualizations saved to {output_folder}")

def evaluate_thresholds(t0_folder, t1_folder, mask_folder, output_csv="threshold_analysis.csv"):
    """
    Evaluate different threshold combinations to help determine optimal values
    """
    # Parameters to test
    mask_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]  # Percentage
    diff_thresholds = [5.0, 10.0, 15.0, 20.0, 30.0]  # Mean pixel difference
    
    results = []
    
    # Get a sample of files (to speed up evaluation)
    t0_files = sorted([f for f in os.listdir(t0_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    t1_files = sorted([f for f in os.listdir(t1_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Choose a smaller subset for faster evaluation
    sample_size = min(100, len(t0_files))
    indices = np.linspace(0, len(t0_files) - 1, sample_size, dtype=int)
    t0_sample = [t0_files[i] for i in indices]
    t1_sample = [t1_files[i] for i in indices]
    mask_sample = [mask_files[i] for i in indices]
    
    # Test each threshold combination
    for mask_thresh in mask_thresholds:
        for diff_thresh in diff_thresholds:
            scene_changes = 0
            
            # Process each frame set
            for t0_file, t1_file, mask_file in zip(t0_sample, t1_sample, mask_sample):
                # Load images
                img_t0 = cv2.imread(os.path.join(t0_folder, t0_file))
                img_t1 = cv2.imread(os.path.join(t1_folder, t1_file))
                mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
                
                # Calculate mask statistics
                mask_pixel_count = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_percentage = (mask_pixel_count / total_pixels) * 100
                
                # Calculate difference between t0 and t1 (only in masked region)
                if mask_pixel_count > 0:
                    gray_t0 = cv2.cvtColor(img_t0, cv2.COLOR_BGR2GRAY)
                    gray_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2GRAY)
                    
                    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    masked_t0 = cv2.bitwise_and(gray_t0, gray_t0, mask=mask_binary)
                    masked_t1 = cv2.bitwise_and(gray_t1, gray_t1, mask=mask_binary)
                    
                    diff = cv2.absdiff(masked_t0, masked_t1)
                    mean_diff = np.sum(diff) / mask_pixel_count if mask_pixel_count > 0 else 0
                else:
                    mean_diff = 0
                
                # Apply current thresholds
                if (mask_percentage > mask_thresh) and (mean_diff > diff_thresh):
                    scene_changes += 1
            
            # Store results for this threshold combination
            results.append({
                'mask_threshold': mask_thresh,
                'diff_threshold': diff_thresh,
                'scene_changes_detected': scene_changes,
                'percentage_detected': (scene_changes / sample_size) * 100
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Threshold analysis saved to {output_csv}")
    return df

# Example usage
if __name__ == "__main__":
    # Replace these with your actual folder paths
    t0_folder = "path/to/t0"
    t1_folder = "path/to/t1"
    mask_folder = "path/to/mask"
    
    # Analyze scene changes and create spreadsheet
    df = analyze_scene_changes(t0_folder, t1_folder, mask_folder)
    
    # Visualize some examples
    visualize_examples(df, t0_folder, t1_folder, mask_folder)
    
    # Optional: evaluate different thresholds
    threshold_df = evaluate_thresholds(t0_folder, t1_folder, mask_folder)