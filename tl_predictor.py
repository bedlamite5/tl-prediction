import numpy as np
import cv2, PIL, os, shutil
from glob import glob

from keras import backend as K
from keras.layers import Input, Conv2D
from keras.models import Model

from yad2k.models.keras_yolo import yolo_body, yolo_eval, yolo_head
from yad2k.utils.draw_boxes import draw_boxes

from PIL import Image
from collections import defaultdict

class TLPredictor(object):

    def __init__(self, num_classes, anchors=[(0.15, 0.47), (0.24, 0.80), (0.40, 1.29), (0.62, 2.09), (0.92, 2.94)], score_threshold=.6):
        self.anchors = anchors
        image_input = Input(shape=(416, 416, 3))

        yolo_model = yolo_body(image_input, len(self.anchors), num_classes)
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
 
        final_layer = Conv2D(len(self.anchors) * (5 + num_classes), (1, 1), activation='linear')(topless_yolo.output)
        self.model_body = Model(image_input, final_layer)

        yolo_outputs = yolo_head(self.model_body.output, self.anchors, num_classes)
        self.input_image_shape = K.placeholder(shape=(2,))
        self.boxes, self.scores, self.classes = yolo_eval(yolo_outputs, self.input_image_shape, score_threshold=score_threshold)

    def predict(self, images, class_names, weights_name):
        self.model_body.load_weights(weights_name)
        sess = K.get_session()
        results = []
        for idx, i_path in enumerate(images):
            im = PIL.Image.fromarray(i_path)
            image_data = np.array(im.resize((416, 416), Image.BICUBIC), dtype=np.float) / 255.
            image_data = np.expand_dims(image_data, 0)
            feed_dict = {self.model_body.input: image_data, self.input_image_shape: [im.size[1], im.size[0]], K.learning_phase(): 0}
            out_boxes, out_scores, out_classes = sess.run([self.boxes, self.scores, self.classes], feed_dict=feed_dict)
            for i, c in list(enumerate(out_classes)):
                box_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                results.append((idx, box_class, score, top, left, bottom, right))
                
        K.clear_session()
        results = sorted(results, key=lambda r:r[4])
        
        return results
    
    def parse_prediction_results(self, image_paths, annotations_dict, results):
        for result in results:
            annotation = {}
            annotation['class'] = result[1]
            annotation['type'] = 'rect'
            annotation['x'] = float(result[4]) 
            annotation['y'] = float(result[3]) 
            annotation['width'] = float(result[6] - result[4]) 
            annotation['height'] = float(result[5] - result[3])
            annotation['confidence'] = float(result[2])
            
            image_path = image_paths[result[0]]
            annotations_dict[image_path].append(annotation)
    
    def remove_duplicates(self, annotations_dict):
        for image_path in annotations_dict:
            dedup = {}
            annotations = annotations_dict[image_path]
            for annotation in annotations:
                cls = annotation['class']
                if cls in dedup and dedup[cls]['confidence'] < annotation['confidence']:
                    dedup[cls] = annotation
                else:
                    dedup[cls] = annotation
    
            annotations = []
            for cls in dedup:
                annotations.append(dedup[cls])
            annotations_dict[image_path] = annotations
            
    # draw boxes
    def draw_predictions(self, image, annotations, class_names=[]):
        boxes = []
        box_classes = []
        scores = []
        
        for a in annotations:
            x = int(a['x'])
            y = int(a['y'])
            w = int(a['width'])
            h = int(a['height'])
            score = a['confidence']
            cls = a['class']
            boxes.append([y, x, (y + h), (x + w)])
            scores.append(score)
            box_classes.append(class_names.index(cls))
                
        image = draw_boxes(image, boxes, box_classes, class_names, scores=np.asarray(scores))
        
        return image
        
if __name__ == '__main__':
    
    MODELS_DIR = "models/"
    UNSEEN_DIR = "unseen/"
    OUTPUT_DIR = "output/"
    
    class_names = ["Red", "Green", "Yellow"]
    
    # cleanup output dir 
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # load unseen images 
    PATTERN = "unseen/**/*.jpg"
    unseen_images = list(glob(PATTERN, recursive=True))
    images = []
    image_paths = []
    for image_path in unseen_images:
        image = cv2.imread(image_path)  # load image
        image_path = image_path.replace(UNSEEN_DIR, "")
        images.append(image)
        image_paths.append(image_path)
    
    # detect tl
    weights_name = MODELS_DIR + "tl_yolo_model.h5"
    predictor = TLPredictor(num_classes=len(class_names))
    results = predictor.predict(images, class_names, weights_name)
    
    # parse results
    annotations_dict = defaultdict(list)
    predictor.parse_prediction_results(image_paths, annotations_dict, results)
    predictor.remove_duplicates(annotations_dict)
    
    # save annotated image
    for image_path in annotations_dict:
        annotations = annotations_dict[image_path]
        
        image = cv2.imread(UNSEEN_DIR + image_path)
        image = predictor.draw_predictions(image, annotations, class_names)
        cv2.imwrite(OUTPUT_DIR + "/" + image_path, image) # save annotated image
        
    print("Done")
