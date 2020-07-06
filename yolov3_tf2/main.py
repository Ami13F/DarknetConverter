import tensorflow as tf
from utils import load_class_names, output_boxes, draw_output, resize_image, transform_images, choose_model
import cv2
import numpy as np
from model import YoloModel
import os.path
from weights import WeightsConvertor

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Main:
    def __init__(self, class_names_file, model_size, model):
        self.class_names_file = class_names_file
        self.model_size = model_size
        self.model = model

    def detect_image(self, image_file, output_file):
        print("Detecting objects loading....")

        img_path = os.path.abspath("yolov3_tf2\\" + image_file)
        class_names = load_class_names(self.class_names_file)

        img_raw = tf.image.decode_image(
            open(img_path, 'rb').read(), channels=3)

        # create one more dimension for batch (1, 416, 416, 3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, self.model_size[0])  # 416 size
        # boxes, classes, scores, nums = model.predict(resized_frame)
        boxes, classes, scores, nums = self.model(img)
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, dsize=(
            self.model_size[0], self.model_size[0]))  # 416 size
        img = draw_output(img, boxes, scores,
                          classes, nums, class_names)
        win_name = 'Detected objects'
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # For saving the result
        cv2.imwrite(os.path.abspath(output_file), img)
        print("Image saved....")

    def convert_model(self, model_name):
        print("Converting model to tflite.....")

        converter = tf.lite.TFLiteConverter.from_saved_model("saved")
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # For old convertor TOCO
        converter.experimental_new_converter = False

        # converter.experimental_new_converter = True

        # # Convert the model
        tflite_model = converter.convert()

        open(model_name, "wb").write(tflite_model)
        print("Tflite model saved")

    def convert_weights(self, yolo, weights_file, weights_dest):
        wc = WeightsConvertor(yolo, weights_file)
        model = wc.model
        try:
            wc.load_weights()
            model.save_weights(weights_dest)
            print("Saved file: " + weights_dest.split("/")[-1])

        except IOError as e:
            print(e)


if __name__ == '__main__':
    print("Running the programm...")

    # Change model type
    model_type = "tiny-prn"

    class_names_file, cfg_file, weights_file, weights_dest = choose_model(
        model_type)

    model_size = (416, 416, 3)
    num_classes = 10
    iou_threshold = 0.3
    confidence_threshold = 0.5

    # Change parameters value for different run modes
    weights_convert = True
    detect_image = False
    model_conversion = True

    yolo = YoloModel(cfg_file, model_size, num_classes,
                     iou_threshold, confidence_threshold)

    model = yolo.create_network()

    main = Main(class_names_file, model_size, model)

    if weights_convert:
        print("Start weights convertor...")
        main.convert_weights(yolo, weights_file, weights_dest)

    if detect_image:
        input_image = "7.jpg"
        test_file = "tiny-prn-3-test.jpg"
        output_image = "yolov3_tf2\\img\\"+test_file
        model.load_weights(weights_dest)
        main.detect_image(input_image, output_image)

    if model_conversion:
        model.load_weights(weights_dest)
        tflite_model_name = "tiny-prn.tflite"
        # save the model as .pb
        print("Start saving model as pb....")
        tf.saved_model.save(model, 'saved')
        # # Create a converter and tflite model
        print("End model conversion as pb....")
        main.convert_model(tflite_model_name)
