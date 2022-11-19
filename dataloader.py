from collections import namedtuple
import cv2
import os
import numpy as np

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def loadImages(x_path, y_path, images_scale = 1/2, number_of_images = 200, skip_images = 0, x_dtype = "float16", y_dtype=np.bool_, parallel =True, threads = 12, flip=False):   
    if parallel == True:
        from multiprocessing import Pool 
    
    img_count = 0
    skiped_images = 0
    x_files = []
    for folder in os.listdir(x_path):
        if img_count == number_of_images:
            break
        data_names = os.listdir(x_path+"/"+folder)
        for data_name in data_names:
            if skiped_images < skip_images:
                skiped_images+=1
                continue 
            x_files.append(x_path+"/"+folder+"/"+data_name)
            img_count+=1
            if img_count == number_of_images:
                break
    x = []
    if parallel == True:
        with Pool(processes=threads) as pool:
            x = pool.starmap(read_image, zip(x_files, np.ones(len(x_files))*images_scale))
    else:
        for file in x_files:
            x.append(read_image(file, image_scale=images_scale))
    
    if x_dtype == np.uint8 or x_dtype == "uint8":
        x = np.array(x, dtype=x_dtype)
    else:
        x = np.array(x, dtype=x_dtype)/255.0

    if flip==True:
        x = np.concatenate((x, np.flip(x, 2)))

    print(f"X shape = {x.shape}.dtype({x_dtype})", "of size", x.nbytes/1e9, "GB")

    img_count = 0
    skiped_images = 0
    y_raw_files = []
    for folder in os.listdir(y_path):
        if img_count == number_of_images:
            break
        data_names = os.listdir(y_path+"/"+folder)
        data_names = [x for x in data_names if "labelIds" in x]
        for data_name in data_names:
            if skiped_images < skip_images:
                skiped_images+=1
                continue 
            y_raw_files.append(y_path+"/"+folder+"/"+data_name)
            img_count+=1
            if img_count == number_of_images:
                break
    y_raw = []
    if parallel == True:
        with Pool(processes=threads) as pool:
            y_raw = pool.starmap(read_image, zip(y_raw_files, np.ones(len(y_raw_files))*images_scale))
    else:
        for file in y_raw_files:
            y_raw.append(read_image(file, image_scale=images_scale))

    y_raw = np.array(y_raw, dtype="uint8")
    y_raw = y_raw[:,:,:,0:1]
    
    category2names = {0:["car", "truck", "bus", "caravan"], 1:["road"]}
    category2ids = []
    for cat in category2names:
        ids = []
        for label in labels:
            if label.name in category2names[cat]:
                ids.append(label.id)
        category2ids.append(ids)
    #append as the last category every that dosent belong in names of category2names
    ids = []
    names = []
    for values in category2names.values():
        for name in values:
            names.append(name)
    for label in labels:
        if label.name not in names:
            ids.append(label.id)
    category2ids.append(ids)

    if y_dtype == "sparce":
        y = []
        if parallel == True:
            with Pool(processes=threads) as pool:
                y = pool.starmap(filter_labels, zip(y_raw, [category2ids for x in range(len(y_raw))]))
        else:
            for image in y_raw:
                y.append(filter_labels(image, category2ids))
        y = np.array(y, dtype=np.uint8)

    #to do
    # if y_dtype != "sparce":
    #     y = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(CarlaTags2ids)), dtype=y_dtype)
    #     for imgIndex, image in enumerate(y_raw[ :, :, :, 0]):
    #         for tag in range(y.shape[-1]):
    #             ids = CarlaTags2ids[tag]
    #             for id in ids:
    #                 y[imgIndex, :, :, tag] += (image == id).astype(y_dtype)
    
    if flip==True:
        y = np.concatenate((y, np.flip(y, 2)))
        
    print(f"Y shape = {y.shape}.dtype(uint8)", "of size", y.nbytes/1e9, "GB")
    return x, y

def labels2RGB(images, avarage = False):
    if avarage == False:
        shape = list(images.shape)
        shape[-1] = 3
        images_rgb = np.zeros(shape=shape, dtype=np.uint8)
        Tags2color =  {0:(0,0,142), 1:(128, 64, 128), 2:(0,0,0)}
        shape[0] = len(Tags2color)
        Tags2colorMatrix = np.zeros(shape=shape, dtype=np.uint8)
        for tag in Tags2color:
            Tags2colorMatrix[tag,:,:,:] = Tags2color[tag]
            
        for imageIndex, image in enumerate(images):
            for tag in Tags2color:
                x = (image == tag)
                x = np.concatenate((x,x,x), axis=2)
                images_rgb[imageIndex]+= x * Tags2colorMatrix[tag,:,:,:]
        return images_rgb

    else:
        shape = list(images.shape)
        shape[-1] = 3
        images_rgb = np.zeros(shape=shape, dtype=np.float32)
        Tags2color =  {0:(0,0,142), 1:(128, 64, 128), 2:(0,0,0)}
        shape[0] = len(Tags2color)
        Tags2colorMatrix = np.zeros(shape=shape, dtype=np.float32)
        for tag in Tags2color:
            Tags2colorMatrix[tag,:,:,:] = Tags2color[tag]

        for imageIndex, image in enumerate(images):
            for tag in Tags2color:
                x = image[:,:,tag:tag+1]
                x = np.concatenate((x,x,x), axis=2)
                images_rgb[imageIndex]+= x * Tags2colorMatrix[tag,:,:,:]
        images_rgb = images_rgb.astype(np.uint8)
        return images_rgb
        
    
def read_image(file, image_scale):
    img  = cv2.imread(file)
    img = cv2.resize(img, (int(img.shape[1]*image_scale), int(img.shape[0]*image_scale)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def filter_labels(image, category2ids):
    shape = list(image.shape)
    shape[-1] = 1
    image_new = np.zeros(shape, dtype=np.uint8)
    for tag, ids in enumerate(category2ids):
        for id in ids:
            image_new += (image==id).astype(np.uint8) * tag
    return image_new