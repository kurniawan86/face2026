import os
import cv2
import numpy as np
from preprocessing import FacePreprocessing


class DatasetLoader:
    def __init__(self, base_path, img_size=(112,112),augment=False, augment_times=1):
        self.augment_times = augment_times
        self.base_path = base_path
        self.preprocess = FacePreprocessing(img_size,augment=augment)
        self.label_map = {}

    def _load_folder(self, folder_path, update_label_map=False, is_train=False):
        X = []
        y = []

        class_names = sorted(os.listdir(folder_path))

        # hanya update label_map saat TRAINING
        if update_label_map:
            for idx, class_name in enumerate(class_names):
                self.label_map[class_name] = idx

        for class_name in class_names:
            class_path = os.path.join(folder_path, class_name)

            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                if is_train and self.preprocess.augment_flag:
                    for _ in range(self.augment_times):
                        aug_img = self.preprocess.process_training(img)
                        X.append(aug_img)
                        y.append(self.label_map[class_name])
                else:
                    processed = self.preprocess.process_inference(img)
                    X.append(processed)
                    y.append(self.label_map[class_name])

        return np.array(X), np.array(y)

    def load_data(self):
        train_path = os.path.join(self.base_path, "TRAINING")
        test_path = os.path.join(self.base_path, "TESTING")

        X_train, y_train = self._load_folder(train_path, True, True)
        X_test, y_test = self._load_folder(test_path, False, False)

        return X_train, X_test, y_train, y_test
    
if __name__=="__main__":
    loader = DatasetLoader(
    "DATASET_FACES",
    augment=True,
    augment_times=3)

    X_train, X_test, y_train, y_test = loader.load_data()

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)
    print("Label map:", loader.label_map)