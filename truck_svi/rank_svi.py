import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# import seaborn as sns  # Optional, using matplotlib instead
from pathlib import Path

# Set up directories
DIR_PROCESSED = "data/processed"
DIR_SVI_IMAGES = os.path.join(DIR_PROCESSED, "svi_images")

def load_svi_data():
    """Load and combine SVI data from both cities"""
    city_list = ["cuttack", "kanpur"]
    all_data = []
    
    for city in city_list:
        csv_path = os.path.join(DIR_SVI_IMAGES, city, "nearest_svi.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['city'] = city
            all_data.append(df)
            print(f"Loaded {len(df)} records from {city}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined records: {len(combined_df)}")
        return combined_df
    else:
        raise FileNotFoundError("No SVI data files found")

def calculate_percentiles(df):
    """Calculate 10th percentiles of deviceSpeed"""
    # Remove any NaN values in deviceSpeed
    df_clean = df.dropna(subset=['deviceSpeed'])
    
    # Calculate percentiles
    percentiles = np.arange(0, 101, 10)
    speed_percentiles = np.percentile(df_clean['deviceSpeed'], percentiles)
    
    # Create percentile bins
    df_clean['speed_percentile_bin'] = pd.cut(
        df_clean['deviceSpeed'], 
        bins=speed_percentiles, 
        labels=[f'{i}-{i+10}%' for i in range(0, 100, 10)],
        include_lowest=True
    )
    
    return df_clean, speed_percentiles

def get_image_paths_for_samples(df, samples_per_bin=10):
    """Find available image paths for sampled records"""
    image_data = []
    
    for bin_name in df['speed_percentile_bin'].cat.categories:
        bin_data = df[df['speed_percentile_bin'] == bin_name]
        
        if len(bin_data) >= samples_per_bin:
            sampled = bin_data.sample(n=samples_per_bin, random_state=42)
        else:
            sampled = bin_data
        
        for _, row in sampled.iterrows():
            # Look for corresponding image files
            city = row['city']
            panoid = row['panoid']
            
            # Check in gsv_panorama folder
            panorama_pattern = f"data/processed/svi_images/{city}/gsv_panorama/*/{panoid}.jpg"
            
            image_files = list(Path(".").glob(panorama_pattern))
            
            if image_files:
                image_data.append({
                    'panoid': panoid,
                    'deviceSpeed': row['deviceSpeed'],
                    'city': city,
                    'percentile_bin': bin_name,
                    'image_path': str(image_files[0])  # Take first available image
                })
    
    return pd.DataFrame(image_data)

def create_speed_analysis_plots(df, speed_percentiles):
    """Create comprehensive speed analysis plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    # sns.set_palette("husl")  # Using matplotlib default colors
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Speed distribution histogram
    plt.subplot(3, 2, 1)
    plt.hist(df['deviceSpeed'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(df['deviceSpeed'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["deviceSpeed"].mean():.2f}')
    plt.axvline(df['deviceSpeed'].median(), color='orange', linestyle='--', 
                label=f'Median: {df["deviceSpeed"].median():.2f}')
    plt.xlabel('Device Speed')
    plt.ylabel('Frequency')
    plt.title('Distribution of Device Speeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speed by city
    plt.subplot(3, 2, 2)
    for city in df['city'].unique():
        city_data = df[df['city'] == city]['deviceSpeed']
        plt.hist(city_data, bins=30, alpha=0.6, label=city, edgecolor='black', density=True)
    plt.xlabel('Device Speed')
    plt.ylabel('Frequency')
    plt.title('Device Speed Distribution by City')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Percentile ranges bar plot
    plt.subplot(3, 2, 3)
    percentile_labels = [f'{i}th' for i in range(0, 101, 10)]
    plt.bar(percentile_labels, speed_percentiles, alpha=0.7, edgecolor='black')
    plt.xlabel('Percentile')
    plt.ylabel('Speed Value')
    plt.title('Speed Values at Each 10th Percentile')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Box plot by percentile bins (using matplotlib)
    plt.subplot(3, 2, 4)
    df_for_box = df.dropna(subset=['speed_percentile_bin'])
    if not df_for_box.empty:
        # Create box plot data manually
        box_data = []
        labels = []
        for bin_name in df_for_box['speed_percentile_bin'].cat.categories:
            bin_speeds = df_for_box[df_for_box['speed_percentile_bin'] == bin_name]['deviceSpeed']
            if len(bin_speeds) > 0:
                box_data.append(bin_speeds.values)
                labels.append(bin_name)
        
        if box_data:
            plt.boxplot(box_data, labels=labels)
            plt.xticks(rotation=45)
            plt.xlabel('Speed Percentile Bin')
            plt.ylabel('Device Speed')
            plt.title('Speed Distribution within Each Percentile Bin')
    
    # Plot 5: Count of samples per percentile bin
    plt.subplot(3, 2, 5)
    bin_counts = df['speed_percentile_bin'].value_counts().sort_index()
    bin_counts.plot(kind='bar', alpha=0.7, edgecolor='black')
    plt.xlabel('Speed Percentile Bin')
    plt.ylabel('Count of Records')
    plt.title('Number of Records per Percentile Bin')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Speed ranges table
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Create speed range summary
    range_data = []
    for i in range(len(speed_percentiles)-1):
        range_data.append([
            f'{i*10}-{(i+1)*10}%',
            f'{speed_percentiles[i]:.2f}',
            f'{speed_percentiles[i+1]:.2f}',
            f'{speed_percentiles[i+1] - speed_percentiles[i]:.2f}'
        ])
    
    table = plt.table(
        cellText=range_data,
        colLabels=['Percentile', 'Min Speed', 'Max Speed', 'Range'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.title('Speed Ranges by Percentile', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "speed_analysis_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined speed analysis plot saved to: {output_path}")
    
    return fig

def create_sample_images_plot(image_df, samples_per_bin=10):
    """Create a plot showing sample images for each percentile bin"""
    
    if image_df.empty:
        print("No images found for visualization")
        return None
    
    # Get unique percentile bins
    bins = sorted(image_df['percentile_bin'].unique())
    n_bins = len(bins)
    
    # Create figure
    fig, axes = plt.subplots(n_bins, samples_per_bin, 
                            figsize=(samples_per_bin * 1, n_bins * 1))
    
    if n_bins == 1:
        axes = axes.reshape(1, -1)
    if samples_per_bin == 1:
        axes = axes.reshape(-1, 1)
    
    for bin_idx, bin_name in enumerate(bins):
        bin_images = image_df[image_df['percentile_bin'] == bin_name]
        
        for img_idx in range(samples_per_bin):
            ax = axes[bin_idx, img_idx]
            
            if img_idx < len(bin_images):
                row = bin_images.iloc[img_idx]
                
                try:
                    # Load and display image
                    img = Image.open(row['image_path'])
                    ax.imshow(img)
                    ax.set_title(f"Speed: {row['deviceSpeed']:.1f}\n{row['city']}", 
                               fontsize=8)
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Image\nNot Found\n{row["panoid"][:8]}...', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.axis('off')
        
        # Add row label
        axes[bin_idx, 0].text(-0.1, 0.5, f'{bin_name}\nPercentile', 
                             rotation=90, ha='center', va='center', 
                             transform=axes[bin_idx, 0].transAxes,
                             fontsize=8, fontweight='bold')
    
    plt.suptitle(f'Sample Street View Images by Speed Percentile\n({samples_per_bin} samples per percentile)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_images_by_percentile_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined sample images plot saved to: {output_path}")
    
    return fig

def create_city_specific_plots(df):
    """Create separate analysis plots for each city"""
    
    for city in df['city'].unique():
        city_df = df[df['city'] == city].copy()
        
        # Calculate city-specific percentiles
        city_df_clean = city_df.dropna(subset=['deviceSpeed'])
        percentiles = np.arange(0, 101, 10)
        city_speed_percentiles = np.percentile(city_df_clean['deviceSpeed'], percentiles)
        
        # Create percentile bins for city
        city_df_clean['speed_percentile_bin'] = pd.cut(
            city_df_clean['deviceSpeed'], 
            bins=city_speed_percentiles, 
            labels=[f'{i}-{i+10}%' for i in range(0, 100, 10)],
            include_lowest=True
        )
        
        # Create city-specific speed analysis plot
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Speed distribution histogram
        plt.subplot(3, 2, 1)
        plt.hist(city_df_clean['deviceSpeed'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(city_df_clean['deviceSpeed'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {city_df_clean["deviceSpeed"].mean():.2f}')
        plt.axvline(city_df_clean['deviceSpeed'].median(), color='orange', linestyle='--', 
                    label=f'Median: {city_df_clean["deviceSpeed"].median():.2f}')
        plt.xlabel('Device Speed')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Device Speeds - {city.title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Percentile ranges bar plot
        plt.subplot(3, 2, 2)
        percentile_labels = [f'{i}th' for i in range(0, 101, 10)]
        plt.bar(percentile_labels, city_speed_percentiles, alpha=0.7, edgecolor='black')
        plt.xlabel('Percentile')
        plt.ylabel('Speed Value')
        plt.title(f'Speed Values at Each 10th Percentile - {city.title()}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Box plot by percentile bins
        plt.subplot(3, 2, 3)
        if not city_df_clean.empty:
            box_data = []
            labels = []
            for bin_name in city_df_clean['speed_percentile_bin'].cat.categories:
                bin_speeds = city_df_clean[city_df_clean['speed_percentile_bin'] == bin_name]['deviceSpeed']
                if len(bin_speeds) > 0:
                    box_data.append(bin_speeds.values)
                    labels.append(bin_name)
            
            if box_data:
                plt.boxplot(box_data, tick_labels=labels)
                plt.xticks(rotation=45)
                plt.xlabel('Speed Percentile Bin')
                plt.ylabel('Device Speed')
                plt.title(f'Speed Distribution within Each Percentile Bin - {city.title()}')
        
        # Plot 4: Count of samples per percentile bin
        plt.subplot(3, 2, 4)
        bin_counts = city_df_clean['speed_percentile_bin'].value_counts().sort_index()
        bin_counts.plot(kind='bar', alpha=0.7, edgecolor='black')
        plt.xlabel('Speed Percentile Bin')
        plt.ylabel('Count of Records')
        plt.title(f'Number of Records per Percentile Bin - {city.title()}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Speed ranges table
        plt.subplot(3, 2, 5)
        plt.axis('off')
        
        # Create speed range summary
        range_data = []
        for i in range(len(city_speed_percentiles)-1):
            range_data.append([
                f'{i*10}-{(i+1)*10}%',
                f'{city_speed_percentiles[i]:.2f}',
                f'{city_speed_percentiles[i+1]:.2f}',
                f'{city_speed_percentiles[i+1] - city_speed_percentiles[i]:.2f}'
            ])
        
        table = plt.table(
            cellText=range_data,
            colLabels=['Percentile', 'Min Speed', 'Max Speed', 'Range'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        plt.title(f'Speed Ranges by Percentile - {city.title()}', pad=20)
        
        # Plot 6: Summary statistics
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        stats_text = f"""
        City: {city.title()}
        
        Total Records: {len(city_df_clean):,}
        
        Speed Statistics:
        • Min: {city_df_clean['deviceSpeed'].min():.2f}
        • Max: {city_df_clean['deviceSpeed'].max():.2f}
        • Mean: {city_df_clean['deviceSpeed'].mean():.2f}
        • Median: {city_df_clean['deviceSpeed'].median():.2f}
        • Std Dev: {city_df_clean['deviceSpeed'].std():.2f}
        
        Percentile Ranges:
        • 0-50th: {city_speed_percentiles[0]:.1f} - {city_speed_percentiles[5]:.1f}
        • 50-90th: {city_speed_percentiles[5]:.1f} - {city_speed_percentiles[9]:.1f}
        • 90-100th: {city_speed_percentiles[9]:.1f} - {city_speed_percentiles[10]:.1f}
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Speed Analysis for {city.title()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save city-specific plot
        output_dir = "reports/figures"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"speed_analysis_{city}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Speed analysis plot for {city} saved to: {output_path}")
        plt.close()
        
        # Create city-specific sample images plot
        print(f"Creating sample images plot for {city}...")
        city_image_df = get_image_paths_for_samples(city_df_clean, samples_per_bin=10)
        
        if not city_image_df.empty:
            create_sample_images_plot_city(city_image_df, city, samples_per_bin=10)
        else:
            print(f"No images found for {city}")

def create_sample_images_plot_city(image_df, city, samples_per_bin=10):
    """Create a plot showing sample images for each percentile bin for a specific city"""
    
    if image_df.empty:
        print(f"No images found for visualization for {city}")
        return None
    
    # Get unique percentile bins
    bins = sorted(image_df['percentile_bin'].unique())
    n_bins = len(bins)
    
    # Create figure
    fig, axes = plt.subplots(n_bins, samples_per_bin, 
                            figsize=(samples_per_bin * 1, n_bins * 1))
    
    if n_bins == 1:
        axes = axes.reshape(1, -1)
    if samples_per_bin == 1:
        axes = axes.reshape(-1, 1)
    
    for bin_idx, bin_name in enumerate(bins):
        bin_images = image_df[image_df['percentile_bin'] == bin_name]
        
        for img_idx in range(samples_per_bin):
            ax = axes[bin_idx, img_idx]
            
            if img_idx < len(bin_images):
                row = bin_images.iloc[img_idx]
                
                try:
                    # Load and display image
                    img = Image.open(row['image_path'])
                    ax.imshow(img)
                    ax.set_title(f"Speed: {row['deviceSpeed']:.1f}", fontsize=8)
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Image\nNot Found\n{row["panoid"][:8]}...', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.axis('off')
        
        # Add row label
        axes[bin_idx, 0].text(-0.1, 0.5, f'{bin_name}\nPercentile', 
                             rotation=90, ha='center', va='center', 
                             transform=axes[bin_idx, 0].transAxes,
                             fontsize=8, fontweight='bold')
    
    plt.suptitle(f'Sample Street View Images by Speed Percentile - {city.title()}\n({samples_per_bin} samples per percentile)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sample_images_by_percentile_{city}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Sample images plot for {city} saved to: {output_path}")
    plt.close()
    
    return fig

def print_speed_summary(df, speed_percentiles):
    """Print detailed speed summary statistics"""
    print("\n" + "="*60)
    print("SPEED RANKING ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Cities: {', '.join(df['city'].unique())}")
    print(f"- Speed range: {df['deviceSpeed'].min():.2f} - {df['deviceSpeed'].max():.2f}")
    print(f"- Mean speed: {df['deviceSpeed'].mean():.2f}")
    print(f"- Median speed: {df['deviceSpeed'].median():.2f}")
    print(f"- Standard deviation: {df['deviceSpeed'].std():.2f}")
    
    print(f"\nRecords per city:")
    for city in df['city'].unique():
        city_count = len(df[df['city'] == city])
        city_mean = df[df['city'] == city]['deviceSpeed'].mean()
        print(f"- {city}: {city_count:,} records (mean speed: {city_mean:.2f})")
    
    print(f"\nSpeed Percentiles (10th intervals):")
    print("-" * 40)
    for i in range(len(speed_percentiles)):
        print(f"{i*10:3d}th percentile: {speed_percentiles[i]:8.2f}")
    
    print(f"\nSpeed Ranges by Percentile:")
    print("-" * 50)
    print(f"{'Percentile':<12} {'Min Speed':<10} {'Max Speed':<10} {'Range':<8}")
    print("-" * 50)
    for i in range(len(speed_percentiles)-1):
        range_val = speed_percentiles[i+1] - speed_percentiles[i]
        print(f"{i*10:2d}-{(i+1)*10:2d}%       {speed_percentiles[i]:8.2f}   "
              f"{speed_percentiles[i+1]:8.2f}   {range_val:6.2f}")

def main():
    """Main function to perform SVI ranking analysis"""
    print("Starting SVI Speed Ranking Analysis...")
    
    try:
        # Load data
        df = load_svi_data()
        
        # Calculate percentiles
        df_with_bins, speed_percentiles = calculate_percentiles(df)
        
        # Print summary
        print_speed_summary(df_with_bins, speed_percentiles)
        
        # Create combined speed analysis plots
        create_speed_analysis_plots(df_with_bins, speed_percentiles)
        
        # Try to create combined sample images plot
        print("\nLooking for sample images...")
        image_df = get_image_paths_for_samples(df_with_bins, samples_per_bin=10)
        
        if not image_df.empty:
            print(f"Found {len(image_df)} images for visualization")
            create_sample_images_plot(image_df)
        else:
            print("No images found - creating plot with speed data only")
        
        # Create city-specific plots
        print("\nCreating city-specific analysis plots...")
        create_city_specific_plots(df_with_bins)
        
        # Save ranked data
        output_csv = "data/processed/ranked_svi_by_speed.csv"
        df_ranked = df_with_bins.sort_values('deviceSpeed', ascending=False)
        df_ranked.to_csv(output_csv, index=False)
        print(f"\nRanked SVI data saved to: {output_csv}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
