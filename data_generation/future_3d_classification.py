import csv
import json
import os


class Future3DClassification:
    def __init__(self, data_path):
        self.base_future_3d_dir = os.path.join(data_path, '3d_front')
        self.future_3d_dir = os.path.join(self.base_future_3d_dir, '3D-FUTURE-scene')
        self.model_info_path = os.path.join(self.future_3d_dir, 'GT', 'model_infos.json')
        self.test_json_path = os.path.join(self.future_3d_dir, 'GT', 'test_set.json')
        self.train_json_path = os.path.join(self.future_3d_dir, 'GT', 'train_set.json')

    def generate(self, count):
        train_output_path, test_output_path = self.get_output_paths()
        with open(self.train_json_path) as f:
            train_data = json.load(f)
            image_file_name = self.get_image_file_names(train_data)
            annotations = self.group_annotations(train_data)
            metadata_path = os.path.join(train_output_path, 'metadata.csv')
            for image_id in annotations:
                image_path = os.path.join(self.future_3d_dir, 'train', 'image', image_file_name[image_id] + '.jpg')
                if os.path.exists(image_path):
                    self.log(image_id, image_path, annotations[image_id], metadata_path)
                else:
                    print('Cant find image {}'.format(image_path))

        with open(self.test_json_path) as f:
            test_data = json.load(f)
            image_file_name = self.get_image_file_names(test_data)
            annotations = self.group_annotations(test_data)
            metadata_path = os.path.join(test_output_path, 'metadata.csv')
            for image_id in annotations:
                image_path = os.path.join(self.future_3d_dir, 'test', 'image', image_file_name[image_id] + '.jpg')
                if os.path.exists(image_path):
                    self.log(image_id, image_path, annotations[image_id], metadata_path)
                else:
                    print('Cant find image {}'.format(image_path))

    def get_output_paths(self):
        output_dir = os.path.join(self.base_future_3d_dir, 'generated_classification')
        os.makedirs(output_dir, exist_ok=True)
        train_output_path = os.path.join(output_dir, 'train')
        os.makedirs(train_output_path, exist_ok=True)
        test_output_path = os.path.join(output_dir, 'test')
        os.makedirs(test_output_path, exist_ok=True)

        return train_output_path, test_output_path

    def get_categories(self, json_data):
        raw_categories = json_data['categories']
        return {x['id']: x['category'] for x in raw_categories}

    def group_annotations(self, json_data):
        result = {}
        annotations = json_data['annotations']
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in result:
                result[image_id] = []

            if annotation['category_id'] not in result[image_id]:
                result[image_id].append(annotation['category_id'])

        return result

    def get_image_file_names(self, json_data):
        images = json_data['images']

        result = {}
        for image in images:
            result[image['id']] = image['file_name']

        return result

    def log(self, image_id, path, categories, metadata_path):
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id', 'path', 'categories'])

        with open(metadata_path, 'a', newline='') as f:
            writer = csv.writer(f)
            categories_string = ""
            for category in categories:
                categories_string += str(category) + ";"

            categories_string = categories_string.strip(";")
            writer.writerow([image_id, path, categories_string])

