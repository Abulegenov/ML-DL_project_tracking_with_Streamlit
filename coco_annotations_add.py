import json, os, shutil, argparse
import copy
from typing import List, Dict, Set

segment_classes = [{'id': 1, 'name': 'class_1'}, {'id': 2, 'name': 'class_2'}, {'id': 3, 'name': 'class_3'}]

parser = argparse.ArgumentParser(description='Define parameters')
parser.add_argument('--divide',  type=str, default = None, help='json_file with all annotations', required = False )
parser.add_argument('--join', default=[], nargs='+', required = False)
parser.add_argument('--images', type=str, default = None, required = False)
parser.add_argument('--json_name', type=str, default = None, required = False)
parser.add_argument('--all', type=str, default = None, required = False)

args = parser.parse_args()



def join_annotations(list_of_annotations: List[str], output_filename: str) -> Dict:
    """Merges multiple COCO-style annotation JSON files into a single JSON file.
    
    Args:
        list_of_annotations (List[str]): List of file paths to annotation JSON files.
        output_filename (str): Name of the output JSON file (without extension).
    
    Returns:
        Dict: Merged JSON content.
    """
    if not list_of_annotations:
        raise ValueError("The list of annotation files is empty.")

    # Ensure all file names have '.json' extension
    list_of_annotations = [
        f if f.lower().endswith('.json') else f"{f}.json"
        for f in list_of_annotations
    ]

    annotations_list = []
    
    for file in list_of_annotations:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        
        try:
            with open(file, 'r', encoding='utf-8') as f:
                annotations_list.append(json.load(f))
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in file: {file}")
    
    # Deep copy first annotation set as the base
    merged_annotations = copy.deepcopy(annotations_list[0])
    
    last_image_id = merged_annotations['images'][-1]['id']
    last_annotation_id = merged_annotations['annotations'][-1]['id']
    
    # Merge all remaining annotation files
    for annotations in annotations_list[1:]:
        for image in annotations['images']:
            image['id'] += last_image_id
            merged_annotations['images'].append(image)

        for annotation in annotations['annotations']:
            annotation['id'] += last_annotation_id
            annotation['image_id'] += last_image_id
            merged_annotations['annotations'].append(annotation)

        last_image_id = merged_annotations['images'][-1]['id']
        last_annotation_id = merged_annotations['annotations'][-1]['id']

    output_path = f"{output_filename}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_annotations, f, indent=4, ensure_ascii=False)
    
    return merged_annotations



def create_json_for_classes(json_file: Dict, image_ids: Set[int], file_name: str) -> Dict:
    """Filters and reassigns image and annotation IDs in a COCO-style JSON file.

    Args:
        json_file (Dict): Original JSON data.
        image_ids (Set[int]): Set of image IDs to retain.
        file_name (str): Output JSON file name (without extension).

    Returns:
        Dict: Filtered and modified JSON content.
    """
    if not json_file or 'images' not in json_file or 'annotations' not in json_file:
        raise ValueError("Invalid JSON structure. Missing 'images' or 'annotations' keys.")

    if not image_ids:
        raise ValueError("The set of image IDs is empty.")

    # Deep copy to avoid modifying original data
    filtered_json = copy.deepcopy(json_file)

    # Filter images and annotations
    filtered_json['images'] = [img for img in filtered_json['images'] if img['id'] in image_ids]
    filtered_json['annotations'] = [anno for anno in filtered_json['annotations'] if anno['image_id'] in image_ids]

    # Create a mapping of old IDs to new sequential IDs
    id_map = {old_id: new_id for new_id, old_id in enumerate(img['id'] for img in filtered_json['images'])}

    # Update IDs based on the new mapping
    for img in filtered_json['images']:
        img['id'] = id_map[img['id']]

    for anno in filtered_json['annotations']:
        anno['image_id'] = id_map[anno['image_id']]

    output_path = f"{file_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_json, f, indent=4, ensure_ascii=False)

    return filtered_json


def train_test_split_segmentation(class_num, json_file):
    n = len(json_file['images'])
    train = int(n*0.8)
    test = n-train
    train_img_ids = []
    test_img_ids = []

    for i in range(0, train):
        train_img_ids.append(json_file['images'][i]['id'])
    for i in range(train, n):
        test_img_ids.append(json_file['images'][i]['id'])

    try:
        os.mkdir(class_num)
    except: 
        print(f'folder already exists, pls rename folder {class_num}')
        return None
    os.mkdir(f'{class_num}/train')
    os.mkdir(f'{class_num}/test')
    for i in json_file['images']:
        if i['id'] in train_img_ids:
          shutil.copy(args.images+'/'+i['file_name'], f'{class_num}/train')
        elif i['id'] in test_img_ids:
            shutil.copy(args.images+'/'+i['file_name'], f'{class_num}/test')

    create_json_for_classes(json_file, train_img_ids, f'{class_num}/train')
    
    create_json_for_classes(json_file, test_img_ids, f'{class_num}/test')
    
    

    


def all_join_divide(list_of_all_files, json_name, final_file_name):
    
    args.images = 'all_photos'
    args.json_name = 'temp/'+json_name

    for i in range(0, len(list_of_all_files)):
        list_of_all_files[i] = 'downloaded_annotations/segmentation/' + list_of_all_files[i]
    list_of_all_files.append('temp/'+json_name)

    join_annotations(list_of_all_files, final_file_name)

