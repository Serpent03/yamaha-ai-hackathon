from xml.etree import ElementTree
from Mask_RCNN.mrcnn.visualize import *
from Mask_RCNN.mrcnn.utils import *
from Mask_RCNN.mrcnn.model import *
from Mask_RCNN.mrcnn.config import *
from os import *
from numpy import *
from numpy import asarray_chkfinite
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import skimage.io

class NutDataset(Dataset):

    className = ""

    def extract_boxes(self, filename):
    # load and parse the file
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box[0].text
            xmin = int(box[5].find('xmin').text)
            ymin = int(box[5].find('ymin').text)
            xmax = int(box[5].find('xmax').text)
            ymax = int(box[5].find('ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            if name == "Type A" or name == "Type B" or name == "Type C":
                boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True): 

        self.add_class("dataset", 1, "Type A")
        self.add_class("dataset", 2, "Type B")
        self.add_class("dataset", 3, "Type C")

        images_dir = dataset_dir + '/Images'
        ann_dir = dataset_dir + '/Ann'

        for filename in listdir(images_dir):
            image_id = filename[:-4]
            # if is_train and int(image_id) >= 5:
            #     continue
            # if not is_train and int(image_id) < 6:
            #     continue
            # if !is_train and int(image_id) >= 150:
			# 	continue
            self.add_image('dataset', image_id=image_id, path=f'{images_dir}/{filename}', annotation=f'{ann_dir}/{image_id}.xml', class_ids=[0, 1, 2, 3])


    # load the masks for an image
    def load_mask(self, image_id):
        # print(self.extract_boxes(self.image_info[5]['annotation']))
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if box[4] == 'Type A':
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('Type A'))
            if box[4] == 'Type B':
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('Type B'))
            if box[4] == 'Type C':
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('Type C'))

        return masks, asarray_chkfinite(class_ids, dtype='int32')

    
    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
    
# def plot_actual_vs_predicted(dataset, model, cfg, n_images=3):
# 	# load image and mask    
#     for i in range(n_images):
#         bCount=0 # for each image
        
#         # load the image and mask
#         image = dataset.load_image(i)
#         mask, _ = dataset.load_mask(i)
#         # convert pixel values (e.g. center)
#         scaled_image = mold_image(image, cfg)
#         # convert image into one sample
#         sample = expand_dims(scaled_image, 0)
#         # make prediction
#         yhat = model.detect(sample, verbose=0)[0]
#         # define subplot
#         pyplot.subplot(n_images, 2, i*2+1)
#         # plot raw pixel data
#         pyplot.imshow(image)
#         pyplot.title('Actual')
#         # plot masks
#         for j in range(mask.shape[2]):
#             pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
#         # get the context for drawing boxes
#         pyplot.subplot(n_images, 2, i*2+2)
#         # plot raw pixel data
#         pyplot.imshow(image)
#         pyplot.title('Predicted')
#         ax = pyplot.gca()
#         # plot each box
#         for box in yhat['rois']:
#             # get coordinates
#             bCount += 1
#             y1, x1, y2, x2 = box
#             width, height = x2 - x1, y2 - y1
#             # create the shape
#             rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#             ax.add_patch(rect)
#             # add the count number
#             plt.text(x1, y1, bCount)
#     # show the figure
#     pyplot.show()
    

def plot_actual_vs_predicted(model, cfg, imID, dataset=None):    
    bCount=0 # for each detected instance
    if type(imID) == int:
        image = dataset.load_image(imID)
    if type(imID) == str:
        image = skimage.io.imread(imID)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]
    print(yhat['class_ids'])
    # get the context for drawing boxes
    # plot raw pixel data
    pyplot.imshow(image)
    pyplot.title('Predicted')
    ax = pyplot.gca()
    # plot each box
    for i in range(len(yhat['rois'])):
        # get coordinates
        box = yhat['rois'][i]
        idNum = yhat['class_ids'][i]
        if idNum == 1:
            name = "Type A"
        if idNum == 2:
            name = "Type B"
        if idNum == 3:
            name = "Type C"
        bCount += 1
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)
        # add the count number
        plt.text(x1, y1, bCount)
        plt.text(x2, y1, name)
# show the figure
    pyplot.show()

train_set = NutDataset()
train_set.load_dataset('Dataset', is_train=True)
train_set.prepare()

test_set = NutDataset()
test_set.load_dataset('Trainset', is_train=False)
test_set.prepare()

# image_id = 6
# image = test_set.load_image(image_id)
# mask, class_ids = test_set.load_mask(image_id)
# bbox = extract_bboxes(mask)
# display_instances(image, bbox, mask, class_ids, test_set.class_names)

class NutConfig(Config):
    NAME = "nut_cfg"
    # Number of classes (background + Nut A/B/C)
    NUM_CLASSES = 1 + 3
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 15

class PredictionConfig(Config):
    NAME = "nut_cfg"
    # number of classes (background + Nut A/B/C)
    NUM_CLASSES = 1 + 3
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# TESTIN2G --

cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_nut_cfg_abc.h5', by_name=True)
plot_actual_vs_predicted(model, cfg, "im.jpg")

# TESTING --

# TRAINING --

# config = NutConfig()
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

# TRAINING --

# imgset = 7
#   # plot raw pixel data
# image = train_set.load_image(imgset)
# pyplot.imshow(image)
# 	# plot all masks
# mask, _ = train_set.load_mask(imgset)
# for j in range(mask.shape[2]):
#     pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
# # show the figure
# pyplot.show()