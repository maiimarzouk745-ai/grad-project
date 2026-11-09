import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path


BASE_PATH = r"C:\Users\yahia\Downloads\arabic_dataset\arabic_dataset\Arabic Dataset"  


GESTURES_FOLDERS = {
    'ana': 'انا',           
    'Bahabk': 'بحبك',   
    'naam': 'نعم' ,
    '3awz':'عاوز',
    'elafdal': 'الافضل',           
    'elsalam3lykm': 'السلام عليكم',   
    'eskot':'اسكت',
    'koleyat handasa': 'كليه الهندسه',           
    'Lawsam7t': 'لو سمحت',   
    'Mas2ol':'مسؤول',
    'sahal': 'سهل',           
    'ta3ban': 'تعبان',   
    'za3lan': 'زعلان',
    'esmy': 'اسمي',
    'Kareem':'كريم',
    'la':'لأ',
    'shokran':'شكرا',
    'akol':'اكل',
    'enta': 'إنت',
    '2awy': 'قوي'
    
         
}

OUTPUT_FILE = "sign_language_dataset.csv"


def load_gesture_files(folder_path, label):
    """
    Load all CSV files for a specific gesture and assign label + gesture ID
    """
    all_data = []
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {folder_path}")
        return None
    
    print(f"\nProcessing folder '{label}': {len(csv_files)} files")
    
    gesture_id = 1
    
    for file_path in sorted(csv_files):
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Required columns
            required_cols = ['timestamp', 'flex_thumb', 'flex_index', 'flex_middle', 
                           'flex_ring', 'flex_pinky', 'ax', 'ay', 'az', 
                           'gx', 'gy', 'gz', 'pitch', 'roll']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Skipping {os.path.basename(file_path)}: missing columns {missing_cols}")
                continue
            
            # Assign gesture ID and label
            df['Gesture_ID'] = gesture_id
            df['Label'] = label
            
            all_data.append(df)
            gesture_id += 1
            
            print(f"  Loaded {os.path.basename(file_path)}: {len(df)} samples")
            
        except Exception as e:
            print(f"  Error in {os.path.basename(file_path)}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Combine all files in this folder
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples for '{label}': {len(combined_df)}")
    
    return combined_df


def create_unified_dataset(base_path, gestures_folders, output_file):
    """
    Merge all gesture datasets into one unified dataset
    """
    
    print("="*60)
    print("Starting dataset merging process")
    print("="*60)
    
    all_gestures_data = []
    gesture_counter = 1 
    
    for folder_name, label in gestures_folders.items():
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        
        # Load gesture data
        gesture_df = load_gesture_files(folder_path, label)
        
        if gesture_df is not None:
            # Re map gesture IDs to be unique
            unique_ids = gesture_df['Gesture_ID'].unique()
            id_mapping = {old_id: gesture_counter + i for i, old_id in enumerate(unique_ids)}
            gesture_df['Gesture_ID'] = gesture_df['Gesture_ID'].map(id_mapping)
            
            gesture_counter += len(unique_ids)
            all_gestures_data.append(gesture_df)
    
    if not all_gestures_data:
        print("\nError: No data found!")
        return
    
    print("\n" + "="*60)
    print("Merging all data...")
    
    final_df = pd.concat(all_gestures_data, ignore_index=True)
    
    # Reorder columns
    column_order = [
        'Gesture_ID', 'timestamp',
        'flex_thumb', 'flex_index', 'flex_middle', 'flex_ring', 'flex_pinky',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'pitch', 'roll',
        'Label'
    ]
    
    final_df = final_df[column_order]
    
    # Save 
    output_path = os.path.join(base_path, output_file)
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("Dataset merged successfully")
    print("="*60)
    print(f"\nSaved file: {output_path}")
    print(f"Total samples: {len(final_df):,}")
    print(f"Unique gestures: {final_df['Gesture_ID'].nunique()}")
    print(f"Number of classes: {final_df['Label'].nunique()}")
    
    print("\nSample distribution per class:")
    label_counts = final_df['Label'].value_counts()
    for label, count in label_counts.items():
        gesture_count = final_df[final_df['Label'] == label]['Gesture_ID'].nunique()
        print(f"  - {label:15s}: {count:5d} samples ({gesture_count} gestures)")
    
    print("\nGesture statistics:")
    gesture_stats = final_df.groupby('Gesture_ID').size()
    print(f"  - Avg samples per gesture: {gesture_stats.mean():.1f}")
    print(f"  - Min samples: {gesture_stats.min()}")
    print(f"  - Max samples: {gesture_stats.max()}")
    
    print("\nPreview (first 5 rows):")
    print(final_df.head())
    
    return final_df


#  Main 
if __name__ == "__main__":
    
    dataset = create_unified_dataset(
        base_path=BASE_PATH,
        gestures_folders=GESTURES_FOLDERS,
        output_file=OUTPUT_FILE
    )
    
    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60)
    