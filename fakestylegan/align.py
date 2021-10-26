# Copyright (c) 2021, Jeremy Fix. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# External modules
import numpy as np
import dlib
import PIL

class Aligner:

    def __init__(self,
                 output_size=1024,
                 transform_size=4096,
                 enable_padding=True,
                 rotate_level=True,
                 random_shift=0.0,
                 retry_crops=False):

        self.output_size = output_size
        self.transform_size = transform_size
        self.enable_padding = enable_padding
        self.rotate_level = rotate_level
        self.random_shift = random_shift
        self.retry_crops = retry_crops

        self.detector = dlib.get_frontal_face_detector()
        # Check if we have the pretrained predictor
        # If not download it
        #TODO

        predictor_model_path = "./shape_predictor_68_face_landmarks.dat"
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    # The following code is adapted from the one released under CC-BY-NC-SA 4.0 
    # by NVIDIA https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    def _align_image(self, img, landmarks):
        """
        img: PIL.Image
        """

        lm = np.array(landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        if self.rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1

        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2

        # Keep drawing new random crop offsets until we find one that is contained in the image
        # and does not require padding
        if self.random_shift != 0:
            for _ in range(1000):
                # Offset the crop rectange center by a random shift proportional to image dimension
                # and the requested standard deviation
                c = (c0 + np.hypot(*x)*2 * self.random_shift * np.random.normal(0, 1, c0.shape))
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
                if not self.retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
                    # We're happy with this crop (either it fits within the image, or retries are disabled)
                    break
            else:
                # rejected N times, give up and move to next image
                # (does not happen in practice with the FFHQ data)
                print('rejected image')
                return

        # Shrink.
        shrink = int(np.floor(qsize / self.output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if self.enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((self.transform_size, self.transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if self.output_size < self.transform_size:
            img = img.resize((self.output_size, self.output_size), PIL.Image.ANTIALIAS)

        return img

    def __call__(self, img):
        """
        img: PIL.Image
        """
        np_img = np.array(img)
        detections = self.detector(np_img, 1)
        if len(detections) == 0:
            raise RuntimeError("No face detected")
        n_detections = len(detections)
        largest_idx = max(zip(range(n_detections), [det.height() * det.width() for det in detections]), key=lambda p:p[1])[0]
  
        selected_detection = detections[largest_idx]
        landmarks = [(item.x, item.y) for item in self.shape_predictor(np_img, selected_detection).parts()]

        return self._align_image(img, landmarks)
