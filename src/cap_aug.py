# coding: utf-8
__author__ = 'RocketFlash: https://github.com/RocketFlash'

import cv2
import numpy as np
import random
from src.utils import resize_keep_ar


class CAP_AUG(object):
    '''
    source_images - list of images paths
    n_objects_range - [min, max] number of objects
    h_range - range of objects heights
              if bev_transform is not None range in meters, else in pixels
    x_range - if bev_transform is None -> range in the image coordinate system (in pixels) [int, int]
              else                     -> range in camera coordinate system (in meters) [float, float]
    y_range - if bev_transform is None -> range in the image coordinate system (in pixels) [int, int]
              else                     -> range in camera coordinate system (in meters) [float, float]
    persons_idxs - objects indexes from dataset to paste [idx1, idx2, ...]
    random_h_flip - random horizontal flip
    coords_format - output coordinates format: [xyxy, xywh, yolo] 
    '''
    def __init__(self, source_images, bev_transform=None, 
                                      n_objects_range=[1, 6],
                                      h_range=[50, 100],
                                      x_range=[200, 500],
                                      y_range=[100 ,300],
                                      persons_idxs=None,
                                      random_h_flip=True,
                                      coords_format='xyxy'):
        
        self.source_images = source_images
        self.bev_transform = bev_transform
        self.n_objects_range = n_objects_range
        self.h_range = h_range
        self.x_range = x_range
        self.y_range = y_range
        self.persons_idxs = persons_idxs
        self.random_h_flip = random_h_flip
        self.coords_format = coords_format


    def __call__(self, image):
        return self.generate_objects(image)


    def generate_objects(self, image):
        n_persons = random.randint(*self.n_objects_range)
        if self.bev_transform is not None:
            points = np.random.uniform(low=[self.x_range[0], self.y_range[0]], 
                                    high=[self.x_range[1], self.y_range[1]], 
                                    size=(n_persons,2))
            heights = np.random.uniform(low=self.h_range[0], 
                                        high=self.h_range[1], 
                                        size=(n_persons,1))
        else:
            points = np.random.randint(low=[self.x_range[0], self.y_range[0]], 
                                    high=[self.x_range[1], self.y_range[1]], 
                                    size=(n_persons,2))
            heights = np.random.randint(low=self.h_range[0], 
                                        high=self.h_range[1], 
                                        size=(n_persons,1))
        
        return self.generate_objects_coord(image, points, heights)


    def generate_objects_coord(self, image, points, heights):
        '''
            points - numpy array of coordinates in meters with shape [n,2]
        '''
        n_persons = points.shape[0]
        
        if self.persons_idxs is None:
            persons_idxs = [random.randint(0,len(self.source_images)-1) for _ in range(n_persons)]
        else:
            persons_idxs = self.persons_idxs
        
        assert len(persons_idxs)==points.shape[0] and points.shape[0]==heights.shape[0]
        
        image_dst = image.copy()
        dst_h, dst_w, _ = image_dst.shape
        coords_all = []
        
        distances = []
        if self.bev_transform is not None:
            points_pixels = self.bev_transform.meters_to_pixels(points)
            distances = self.bev_transform.calculate_dist_meters(points)
            d_sorted_idxs = np.argsort(distances)[::-1]
            distances = distances[d_sorted_idxs]
            heights = heights[d_sorted_idxs]
            points = points_pixels[d_sorted_idxs]
        

        for idx, person_idx in enumerate(persons_idxs):
            point = points[idx]
            height = heights[idx]

            image_src = self.select_images(self.source_images, person_idx)
            x_coord, y_coord = int(point[0]), int(point[1])
            
            if self.bev_transform is not None:
                distance = distances[idx]
                height_pixels = self.bev_transform.get_height_in_pixels(height, distance)
                image_src = resize_keep_ar(image_src, height=height_pixels)
            else:
                image_src = resize_keep_ar(image_src, height=height)
            image_dst, coords = self.paste_object(image_dst, image_src, x_coord, y_coord, self.random_h_flip)
            if coords: coords_all.append(coords)
        
        coords_all = np.array(coords_all)

        if self.coords_format == 'yolo':
            x = coords_all.copy()
            x = x.astype(float)
            dw = 1./dst_w
            dh = 1./dst_h
            ws = (coords_all[:,2] - coords_all[:,0])
            hs = (coords_all[:,3] - coords_all[:,1])
            x[:,0] = dw * (coords_all[:,0] + ws/2.0)
            x[:,1] = dh * (coords_all[:,1] + hs/2.0)
            x[:,2] = dw * ws
            x[:,3] = dh * hs
            coords_all = x
        elif self.coords_format =='xywh':
            x = coords_all.copy()
            x[:,2] = (coords_all[:,1] - coords_all[:,0])
            x[:,3] = (coords_all[:,3] - coords_all[:,2])
            coords_all = x

        return image_dst, coords_all


    def select_images(self, source_images, person_idx):
        source_image_path = source_images[person_idx]
        image_src = cv2.imread(str(source_image_path), cv2.IMREAD_UNCHANGED)
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2BGRA)
        return image_src


    def paste_object(self, image_dst, image_src, x_coord, y_coord, random_h_flip=True):
        src_h, src_w, _ = image_src.shape
        dst_h, dst_w, _ = image_dst.shape
        x_offset, y_offset = x_coord-int(src_w/2), y_coord-src_h
        y1, y2 = max(y_offset, 0), min(y_offset + src_h, dst_h)
        x1, x2 = max(x_offset, 0), min(x_offset + src_w, dst_w)
        y1_m = 0 if y1>0 else -y_offset
        x1_m = 0 if x1>0 else -x_offset
        y2_m = src_h if y2<dst_h-1 else dst_h - y_offset 
        x2_m = src_w if x2<dst_w-1 else dst_w - x_offset
        coords = []
        
        if y1_m>=src_h or x1_m>=src_w or y2_m<0 or x2_m<0:
            return image_dst, coords
        
        if random_h_flip:
            if random.uniform(0, 1)>0.5:
                image_src = cv2.flip(image_src, 1)

        # Simple cut and paste without preprocessing
        mask_src = image_src[:,:,3]
        rgb_img =  image_src[:,:,:3]
        mask_inv = cv2.bitwise_not(mask_src)
        img1_bg = cv2.bitwise_and(image_dst[y1:y2, x1:x2],image_dst[y1:y2, x1:x2],mask=mask_inv[y1_m:y2_m, x1_m:x2_m])
        img2_fg = cv2.bitwise_and(rgb_img[y1_m:y2_m, x1_m:x2_m],rgb_img[y1_m:y2_m, x1_m:x2_m],mask=mask_src[y1_m:y2_m, x1_m:x2_m])
        out_img = cv2.add(img1_bg,img2_fg)

        image_dst[y1:y2, x1:x2] = out_img
        coords = [x1,y1,x2,y2]
        
        # Poisson editing
        # kernel = np.ones((5,5),np.uint8)
        # mask_src = cv2.dilate(mask_src, kernel,iterations=2)
        # src_h, src_w, _ = image_src.shape
        # center = (x1+int((x2-x1)/2)), (y1+int((y2-y1)/2))
        # image_dst = cv2.seamlessClone(image_src, image_dst, mask_src, center, cv2.MONOCHROME_TRANSFER )

        
        return image_dst, coords