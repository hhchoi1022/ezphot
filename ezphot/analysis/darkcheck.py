
#%%
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path

#%%
target_pathlist = sorted(glob.glob('/home/hhchoi1022/ezphot/data/mcalibdata/gppy/7DT/7DT_C361K_HIGH_1x1/7DT11/dark/100-*.fits'))
# %%
from ezphot.imageobjects import MasterImage
target_imglist = [MasterImage(path) for path in target_pathlist]
#%%
def detect_hot_pixels(data, threshold_percentile=99.9, min_value=None):
    """
    Detect hot pixels in dark image data.
    
    Parameters
    ----------
    data : numpy.ndarray
        2D image data
    threshold_percentile : float
        Percentile threshold for hot pixel detection (default 99.5%)
    min_value : float, optional
        Minimum value to consider as hot pixel
        
    Returns
    -------
    hot_pixels : list
        List of (y, x) coordinates of hot pixels
    """
    if len(data.shape) != 2:
        return []
    
    # Calculate threshold
    if min_value is None:
        threshold = np.percentile(data, threshold_percentile)
    else:
        threshold = max(min_value, np.percentile(data, threshold_percentile))
    
    # Find hot pixels
    hot_mask = data > threshold
    hot_y, hot_x = np.where(hot_mask)
    
    return list(zip(hot_y, hot_x))
#%%
#%%
results = []

for i in range(len(target_imglist) - 1):
    img1, img2 = target_imglist[i], target_imglist[i+1]
    
    hot1 = set(detect_hot_pixels(img1.data, threshold_percentile=99.9))
    hot2 = set(detect_hot_pixels(img2.data, threshold_percentile=99.9))
    
    common = hot1 & hot2
    appeared = hot2 - hot1
    disappeared = hot1 - hot2
    
    print(f'Date {img2.path.name} - {img1.path.name}: Common = {len(common)}, Appeared = {len(appeared)}, Disappeared = {len(disappeared)}')


#%%
def analyze_hot_pixel_patterns(image_list, threshold_percentile=99.5, min_hot_pixels=5):
    """
    Analyze hot pixel patterns to detect camera changes.
    
    Parameters
    ----------
    image_list : list
        List of MasterImage objects containing dark frames
    threshold_percentile : float
        Percentile threshold for hot pixel detection
    min_hot_pixels : int
        Minimum number of hot pixels required to consider an image
        
    Returns
    -------
    results : dict
        Dictionary containing hot pixel analysis results
    """
    results = {
        'images': [],
        'hot_pixel_maps': {},
        'camera_signatures': {},
        'change_points': []
    }
    
    print(f"Analyzing hot pixel patterns in {len(image_list)} dark images...")
    
    for i, img in enumerate(image_list):
        print(f"Processing image {i+1}/{len(image_list)}: {img.path.name}")
        
        # Load image data
        data = img.data
        
        if data is None or len(data.shape) != 2:
            print(f"Warning: Could not load 2D data for {img.path.name}")
            continue
        
        # Detect hot pixels
        hot_pixels = detect_hot_pixels(data, threshold_percentile)
        
        if len(hot_pixels) < min_hot_pixels:
            print(f"Warning: Too few hot pixels ({len(hot_pixels)}) in {img.path.name}")
            continue
        
        # Create hot pixel signature (sorted coordinates)
        hot_pixel_signature = tuple(sorted(hot_pixels))
        
        # Store results
        image_info = {
            'filename': img.path.name,
            'filepath': str(img.path),
            'index': i,
            'hot_pixels': hot_pixels,
            'hot_pixel_count': len(hot_pixels),
            'signature': hot_pixel_signature,
            'image_shape': data.shape
        }
        
        results['images'].append(image_info)
        results['hot_pixel_maps'][img.path.name] = hot_pixels
        
        # Group by camera signature
        if hot_pixel_signature not in results['camera_signatures']:
            results['camera_signatures'][hot_pixel_signature] = []
        results['camera_signatures'][hot_pixel_signature].append(image_info)
    
    # Detect change points
    results['change_points'] = _detect_camera_changes_from_signatures(results['images'])
    
    return results

def _detect_camera_changes_from_signatures(images):
    """Detect camera changes based on hot pixel signature changes."""
    change_points = []
    
    for i in range(1, len(images)):
        if images[i]['signature'] != images[i-1]['signature']:
            change_points.append({
                'index': i,
                'filename': images[i]['filename'],
                'from_signature': images[i-1]['signature'],
                'to_signature': images[i]['signature']
            })
    
    return change_points

def analyze_dark_patterns(image_list, sample_size=1000):
    """
    Analyze dark image patterns to detect camera changes.
    
    Parameters
    ----------
    image_list : list
        List of MasterImage objects containing dark frames
    sample_size : int
        Number of pixels to sample from each image for analysis
        
    Returns
    -------
    features_df : pandas.DataFrame
        DataFrame containing extracted features for each image
    """
    features = []
    
    print(f"Analyzing {len(image_list)} dark images...")
    
    for i, img in enumerate(image_list):
        print(f"Processing image {i+1}/{len(image_list)}: {img.path.name}")
        
        # Load image data
        data = img.data
        
        if data is None:
            print(f"Warning: Could not load data for {img.path.name}")
            continue
            
        # Remove any NaN or infinite values
        data_clean = data[~np.isnan(data) & np.isfinite(data)]
        
        if len(data_clean) == 0:
            print(f"Warning: No valid data in {img.path.name}")
            continue
        
        # Sample random pixels for analysis
        if len(data_clean) > sample_size:
            sampled_data = np.random.choice(data_clean, sample_size, replace=False)
        else:
            sampled_data = data_clean
        
        # Extract statistical features
        features_dict = {
            'filename': img.path.name,
            'filepath': str(img.path),
            'mean': np.mean(sampled_data),
            'std': np.std(sampled_data),
            'median': np.median(sampled_data),
            'min': np.min(sampled_data),
            'max': np.max(sampled_data),
            'q25': np.percentile(sampled_data, 25),
            'q75': np.percentile(sampled_data, 75),
            'iqr': np.percentile(sampled_data, 75) - np.percentile(sampled_data, 25),
            'skewness': _calculate_skewness(sampled_data),
            'kurtosis': _calculate_kurtosis(sampled_data),
            'shape_x': data.shape[1] if len(data.shape) > 1 else 1,
            'shape_y': data.shape[0] if len(data.shape) > 1 else 1,
            'total_pixels': len(data_clean)
        }
        
        # Analyze spatial patterns (if image is 2D)
        if len(data.shape) == 2:
            spatial_features = _analyze_spatial_patterns(data)
            features_dict.update(spatial_features)
        
        features.append(features_dict)
    
    return pd.DataFrame(features)

def _calculate_skewness(data):
    """Calculate skewness of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def _calculate_kurtosis(data):
    """Calculate kurtosis of the data."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def _analyze_spatial_patterns(data):
    """Analyze spatial patterns in 2D image data."""
    features = {}
    
    # Calculate row and column means
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    
    # Check for systematic patterns
    features['row_mean_std'] = np.std(row_means)
    features['col_mean_std'] = np.std(col_means)
    features['row_mean_range'] = np.max(row_means) - np.min(row_means)
    features['col_mean_range'] = np.max(col_means) - np.min(col_means)
    
    # Check for gradient patterns
    features['row_gradient'] = np.corrcoef(np.arange(len(row_means)), row_means)[0, 1] if len(row_means) > 1 else 0
    features['col_gradient'] = np.corrcoef(np.arange(len(col_means)), col_means)[0, 1] if len(col_means) > 1 else 0
    
    # Analyze corner vs center differences
    h, w = data.shape
    center_h, center_w = h // 2, w // 2
    corner_size = min(h, w) // 4
    
    # Define regions
    corners = [
        data[:corner_size, :corner_size],  # top-left
        data[:corner_size, -corner_size:],  # top-right
        data[-corner_size:, :corner_size],  # bottom-left
        data[-corner_size:, -corner_size:]  # bottom-right
    ]
    center = data[center_h-corner_size:center_h+corner_size, 
                  center_w-corner_size:center_w+corner_size]
    
    corner_means = [np.mean(corner) for corner in corners]
    center_mean = np.mean(center)
    
    features['corner_std'] = np.std(corner_means)
    features['center_corner_diff'] = center_mean - np.mean(corner_means)
    
    return features

def detect_camera_changes(features_df, n_clusters=None, method='auto'):
    """
    Detect camera changes using clustering analysis.
    
    Parameters
    ----------
    features_df : pandas.DataFrame
        DataFrame containing image features
    n_clusters : int, optional
        Number of clusters to use. If None, will be determined automatically
    method : str
        Method to use: 'auto', 'kmeans', 'pca'
        
    Returns
    -------
    results : dict
        Dictionary containing clustering results and change points
    """
    
    # Select numerical features for clustering
    feature_columns = [col for col in features_df.columns 
                      if col not in ['filename', 'filepath'] and 
                      pd.api.types.is_numeric_dtype(features_df[col])]
    
    X = features_df[feature_columns].values
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        n_clusters = _determine_optimal_clusters(X_scaled)
    
    # Perform clustering
    if method == 'kmeans' or method == 'auto':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
    else:
        # Use PCA for dimensionality reduction first
        pca = PCA(n_components=min(5, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_pca)
    
    # Find change points
    change_points = _find_change_points(cluster_labels)
    
    # Create results
    results = {
        'features_df': features_df.copy(),
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'change_points': change_points,
        'feature_columns': feature_columns,
        'scaler': scaler
    }
    
    # Add cluster information to dataframe
    results['features_df']['cluster'] = cluster_labels
    
    return results

def _determine_optimal_clusters(X, max_clusters=10):
    """Determine optimal number of clusters using elbow method."""
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    K_range = range(2, min(max_clusters + 1, len(X)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find the k with highest silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def _find_change_points(cluster_labels):
    """Find points where cluster changes occur."""
    change_points = []
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i-1]:
            change_points.append(i)
    return change_points

def visualize_hot_pixel_analysis(hot_pixel_results, save_path=None, max_images_to_show=5):
    """
    Visualize hot pixel analysis results.
    
    Parameters
    ----------
    hot_pixel_results : dict
        Results from analyze_hot_pixel_patterns function
    save_path : str, optional
        Path to save the plot
    max_images_to_show : int
        Maximum number of images to show in detail
    """
    
    images = hot_pixel_results['images']
    change_points = hot_pixel_results['change_points']
    camera_signatures = hot_pixel_results['camera_signatures']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hot pixel count over time
    indices = [img['index'] for img in images]
    hot_counts = [img['hot_pixel_count'] for img in images]
    
    axes[0, 0].plot(indices, hot_counts, 'o-', alpha=0.7)
    axes[0, 0].set_xlabel('Image Index')
    axes[0, 0].set_ylabel('Number of Hot Pixels')
    axes[0, 0].set_title('Hot Pixel Count Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark change points
    for cp in change_points:
        axes[0, 0].axvline(x=cp['index'], color='red', linestyle='--', alpha=0.7)
        axes[0, 0].text(cp['index'], max(hot_counts), f"Camera Change\n{cp['filename']}", 
                       rotation=90, ha='right', va='top', fontsize=8)
    
    # Plot 2: Camera signature distribution
    signature_counts = [len(imgs) for imgs in camera_signatures.values()]
    signature_labels = [f"Camera {i+1}" for i in range(len(camera_signatures))]
    
    axes[0, 1].bar(signature_labels, signature_counts, alpha=0.7)
    axes[0, 1].set_xlabel('Camera Signature')
    axes[0, 1].set_ylabel('Number of Images')
    axes[0, 1].set_title('Images per Camera')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Hot pixel spatial distribution (first few images)
    if len(images) > 0:
        # Show hot pixel patterns for first few images
        n_show = min(max_images_to_show, len(images))
        for i in range(n_show):
            img_info = images[i]
            hot_pixels = img_info['hot_pixels']
            if len(hot_pixels) > 0:
                y_coords, x_coords = zip(*hot_pixels)
                axes[1, 0].scatter(x_coords, y_coords, alpha=0.6, s=1, 
                                 label=f"{img_info['filename'][:20]}...")
        
        axes[1, 0].set_xlabel('X Coordinate (pixels)')
        axes[1, 0].set_ylabel('Y Coordinate (pixels)')
        axes[1, 0].set_title('Hot Pixel Locations (First Few Images)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Signature stability analysis
    if len(images) > 1:
        # Calculate signature stability (how many consecutive images have same signature)
        stability = []
        current_signature = images[0]['signature']
        current_count = 1
        
        for i in range(1, len(images)):
            if images[i]['signature'] == current_signature:
                current_count += 1
            else:
                stability.append(current_count)
                current_signature = images[i]['signature']
                current_count = 1
        stability.append(current_count)  # Add the last group
        
        axes[1, 1].hist(stability, bins=range(1, max(stability)+2), alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Consecutive Images with Same Signature')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Camera Signature Stability')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hot pixel analysis plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed summary
    print(f"\nHot Pixel Camera Change Detection Summary:")
    print(f"Total images analyzed: {len(images)}")
    print(f"Number of unique camera signatures: {len(camera_signatures)}")
    print(f"Number of camera changes detected: {len(change_points)}")
    
    if change_points:
        print(f"\nCamera Change Points:")
        for i, cp in enumerate(change_points):
            print(f"  {i+1}. Image {cp['index']}: {cp['filename']}")
            print(f"     Hot pixels: {len(cp['from_signature'])} → {len(cp['to_signature'])}")
    
    print(f"\nCamera Signature Details:")
    for i, (signature, imgs) in enumerate(camera_signatures.items()):
        print(f"  Camera {i+1}: {len(imgs)} images")
        print(f"    Hot pixel count: {len(signature)}")
        print(f"    Image range: {imgs[0]['filename']} to {imgs[-1]['filename']}")
        if len(imgs) > 1:
            print(f"    Date range: {imgs[0]['index']} to {imgs[-1]['index']}")

def visualize_camera_changes(results, save_path=None):
    """
    Visualize camera change detection results.
    
    Parameters
    ----------
    results : dict
        Results from detect_camera_changes function
    save_path : str, optional
        Path to save the plot
    """
    
    features_df = results['features_df']
    cluster_labels = results['cluster_labels']
    change_points = results['change_points']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Cluster assignment over time
    axes[0, 0].scatter(range(len(cluster_labels)), cluster_labels, 
                      c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_xlabel('Image Index')
    axes[0, 0].set_ylabel('Cluster')
    axes[0, 0].set_title('Camera Clusters Over Time')
    
    # Mark change points
    for cp in change_points:
        axes[0, 0].axvline(x=cp, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: Mean vs Standard deviation
    scatter = axes[0, 1].scatter(features_df['mean'], features_df['std'], 
                                c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0, 1].set_xlabel('Mean Dark Current')
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Mean vs Std by Cluster')
    plt.colorbar(scatter, ax=axes[0, 1])
    
    # Plot 3: Image dimensions
    if 'shape_x' in features_df.columns and 'shape_y' in features_df.columns:
        axes[1, 0].scatter(features_df['shape_x'], features_df['shape_y'], 
                          c=cluster_labels, cmap='tab10', alpha=0.7)
        axes[1, 0].set_xlabel('Image Width (pixels)')
        axes[1, 0].set_ylabel('Image Height (pixels)')
        axes[1, 0].set_title('Image Dimensions by Cluster')
    
    # Plot 4: Feature importance (if PCA was used)
    if 'pca' in results:
        pca = results['pca']
        feature_importance = np.abs(pca.components_[0])
        feature_names = results['feature_columns']
        axes[1, 1].bar(range(len(feature_importance)), feature_importance)
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('PCA Component 1 Loading')
        axes[1, 1].set_title('Feature Importance')
        axes[1, 1].set_xticks(range(len(feature_names)))
        axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\nCamera Change Detection Summary:")
    print(f"Total images analyzed: {len(features_df)}")
    print(f"Number of camera clusters detected: {results['n_clusters']}")
    print(f"Number of camera changes detected: {len(change_points)}")
    
    if change_points:
        print(f"Change points (image indices): {change_points}")
        print(f"Change points (filenames):")
        for cp in change_points:
            print(f"  - {features_df.iloc[cp]['filename']}")
    
    # Print cluster statistics
    print(f"\nCluster Statistics:")
    for cluster_id in range(results['n_clusters']):
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        print(f"Cluster {cluster_id}: {len(cluster_data)} images")
        if len(cluster_data) > 0:
            print(f"  Mean dark current: {cluster_data['mean'].mean():.2f} ± {cluster_data['mean'].std():.2f}")
            print(f"  Image dimensions: {cluster_data['shape_x'].iloc[0]}x{cluster_data['shape_y'].iloc[0]}")

# %%
# Method 1: Hot Pixel Analysis (Recommended)
print("="*60)
print("METHOD 1: HOT PIXEL ANALYSIS")
print("="*60)
print("Starting hot pixel pattern analysis...")
hot_pixel_results = analyze_hot_pixel_patterns(target_imglist, threshold_percentile=99.5, min_hot_pixels=5)

# Visualize hot pixel results
print("\nVisualizing hot pixel analysis results...")
visualize_hot_pixel_analysis(hot_pixel_results, save_path='hot_pixel_camera_analysis.png')

# %%
# Method 2: Statistical Pattern Analysis (Alternative)
print("\n" + "="*60)
print("METHOD 2: STATISTICAL PATTERN ANALYSIS")
print("="*60)
print("Starting statistical dark pattern analysis...")
features_df = analyze_dark_patterns(target_imglist)

# Detect camera changes using clustering
print("\nDetecting camera changes using clustering...")
results = detect_camera_changes(features_df)

# Visualize statistical results
print("\nVisualizing statistical analysis results...")
visualize_camera_changes(results, save_path='statistical_camera_analysis.png')

# %%
# Compare both methods
print("\n" + "="*60)
print("COMPARISON OF METHODS")
print("="*60)

print(f"Hot Pixel Method:")
print(f"  - Camera changes detected: {len(hot_pixel_results['change_points'])}")
print(f"  - Unique camera signatures: {len(hot_pixel_results['camera_signatures'])}")

print(f"\nStatistical Method:")
print(f"  - Camera changes detected: {len(results['change_points'])}")
print(f"  - Camera clusters: {results['n_clusters']}")

print(f"\nRecommendation: Use Hot Pixel Method for most reliable results!")
print("Hot pixels are physical defects that remain fixed for each camera,")
print("making them the most reliable indicator of camera identity.")

# %%
