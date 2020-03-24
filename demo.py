#coding=utf-8
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

DATASET="car"
MODEL_FILE=DATASET+"/frozen_inference_graph.pb"
PATH_TO_LABELS = os.path.join(DATASET, DATASET+'_label_map.pbtxt')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
NUM_CLASSES =len(label_map.item)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_graph(pbpath):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pbpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_objects_dict_on_image(image, sess, tensor_dict, image_tensor):
    # Run inference
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict

def cal_area( box ):
    return (box[2]-box[0])*(box[3]-box[1])

def get_car_dict(dict):
    car_index = []
    
    for i in range(len(dict['detection_classes'])):
        if dict['detection_classes'][i] == 3:
            car_index.append(i)

    if len(car_index) == 0:
        return None

    dict['detection_boxes'] = [ dict['detection_boxes'][i] for i in car_index ]

    max_area = 0
    index = None    
    for i in range(len(car_index)):
        area = cal_area(dict['detection_boxes'][i])
        if max_area < area:
            max_area = area
            index = i

    dict['num_detections'] = 1
    dict['detection_scores'] = np.array([dict['detection_scores'][car_index[index]]], dtype='float32')
    dict['detection_boxes'] = np.array([ dict['detection_boxes'][index] ])
    dict['detection_classes'] = np.array([3], dtype='uint8' )
    return dict

def guide_usr_align_car_on_image(box, image):
    #x1,y1,x2,y2
    print(box)
    # H = image.shape[0]
    # W = image.shape[1]

    # cv2.circle(image, (int(box[1]*W), int(box[0]*H)), 2, (0,0,255), thickness=2 )
    # cv2.circle(image, (int(box[3]*W), int(box[2]*H)), 2, (0,0,255), thickness=2 )
    # cv2.line(image, (int(box[1]*W), int(box[0]*H)), (int(box[3]*W), int(box[2]*H)), (0,0,255), 2 )
    
    org = (10, 35)
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 1
    color = (0,0,255)
    thickness = 2

    coordinates_status = []

    if box is None:
        image = cv2.putText(image,"No car found",org,
            fontFace, fontScale, color, thickness, cv2.LINE_AA) 
        return

    if box[0] < 0.02:
        coordinates_status.append('up')
    if box[1] < 0.02:
        coordinates_status.append('left')
    if box[2] > 0.98:
        coordinates_status.append('down')
    if box[3] > 0.98:
        coordinates_status.append('right')

    if len(coordinates_status) == 0:
        image = cv2.putText(image,'Perfect',org,
            fontFace, fontScale, color, thickness, cv2.LINE_AA)
    elif len(coordinates_status) == 1:
        image = cv2.putText(image,f'Move your screen {coordinates_status[0]} please',org,
            fontFace, fontScale, color, thickness, cv2.LINE_AA) 
    elif len(coordinates_status) == 2:
        image = cv2.putText(image,f'Move your screen {coordinates_status[0]} and {coordinates_status[1]} please',org,
            fontFace, fontScale, color, thickness, cv2.LINE_AA) 
    else:
        image = cv2.putText(image,f'Zoom out or go far away please',org,
            fontFace, fontScale, color, thickness, cv2.LINE_AA) 

def get_drawed_image(image, sess, tensor_dict, image_tensor):
    #image = load_image_into_numpy_array(Image.open(image_path))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    output_dict = get_objects_dict_on_image(image, sess, tensor_dict, image_tensor)
    output_dict = get_car_dict(output_dict)

    if output_dict is not None:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            min_score_thresh=.1,
            line_thickness=8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        guide_usr_align_car_on_image(output_dict['detection_boxes'][0], image)
        return image

        # cv2.imwrite(image_path.replace('test_images/', '') ,cv2.cvtColor(image, cv2.COLOR_RGB2BGR) )
        # print('Save the ', image_path.replace('test_images/', ''))
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        guide_usr_align_car_on_image(output_dict['detection_boxes'][0], image)
        return image

def app(video_path, graph):
    # capture = cv2.VideoCapture(video_path)
    # _, image = capture.read()

    # writer = cv2.VideoWriter('video.avi',
    #     cv2.VideoWriter_fourcc(*'XVID'), 
    #     fps=30.0, 
    #     frameSize=(640,480))

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks( detection_masks, detection_boxes, 
                                                                                        image.shape[0], image.shape[1] )
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            # cnt = 0
            # while capture.isOpened():
            #     _, image = capture.read()

            #     if _ is False:
            #         print("End Video")
            #         capture.release()
            #         writer.release()
            #         break

            #     cnt += 1
            #     if cnt % 2 == 0:
            #         continue
                
            #     image = get_drawed_image(image, sess, tensor_dict, image_tensor)
            #     cv2.imwrite(f'./result/{cnt}.jpg', image)
                # writer.write(image)
            
            for file in os.listdir('test_images'):
                file_path = 'test_images/'+file
                print(file_path)
                image = cv2.imread(file_path)
                # print(image)
                image = get_drawed_image(image, sess, tensor_dict, image_tensor)
                cv2.imwrite(file, image)

if __name__=="__main__":
    graph=load_graph(MODEL_FILE)
    video_path = './test_images/VID.mp4'
    app(video_path, graph)
