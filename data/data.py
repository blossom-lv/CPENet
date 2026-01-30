import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
#def cv_random_flip(img, label, depth):
def cv_random_flip(img, label, depth, mask):#add_revise
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(len(mask)):#add
            mask[i] = mask[i].transpose(Image.FLIP_LEFT_RIGHT)#add
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    #return img, label, depth
    return img, label, depth, mask#add_revise


#def randomCrop(image, label, depth):
def randomCrop(image, label, depth, mask):#add_revise
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    for i in range(len(mask)):#add
        mask[i] = mask[i].crop(random_region)#add
    #return image.crop(random_region), label.crop(random_region), depth.crop(random_region)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), mask#add_revise


#def randomRotation(image, label, depth):
def randomRotation(image, label, depth, mask):#add_revise
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        for i in range(len(mask)):#add
            mask[i] = mask[i].rotate(random_angle, mode)#add
    #return image, label, depth
    return image, label, depth, mask#add_revise


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    #def __init__(self, image_root, gt_root, depth_root, trainsize):
    def __init__(self, image_root, gt_root, depth_root, masks_root, trainsize):#add_revise
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]

        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png') or f.endswith('.jpg')]
        self.masks = [masks_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]#add
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.masks = sorted(self.masks)#add
        self.filter_files()#add->revise
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])
        self.masks_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])#add


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        mask = self.binary_masks_loader(self.masks[index])#add

        #image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth, mask = cv_random_flip(image, gt, depth, mask)#add->revise
        #image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth, mask = randomCrop(image, gt, depth, mask)#add->revise
        #image, gt, depth = randomRotation(image, gt, depth)
        image, gt, depth, mask = randomRotation(image, gt, depth, mask)#add->revise
        
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        for i in range(len(mask)):#add
            mask[i]  = self.masks_transform(mask[i])#add

        #return image, gt, depth
        return image, gt, depth, mask#add_revise

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.depths) and len(self.depths) == len(self.masks)#add_revise
        images = []
        gts = []
        depths = []
        masks = []#add
        #for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
        for img_path, gt_path, depth_path, masks_path in zip(self.images, self.gts, self.depths, self.masks):#add_revise
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            mask = []
            for i in range(4):#add
                img_name = os.path.splitext(os.path.basename(masks_path))[0]#add
                mask.append(Image.open(os.path.join(os.path.dirname(masks_path), img_name + '_ersion_img_' + str(i) + '.png')))#add
            #if img.size == gt.size and gt.size == depth.size:
            if img.size == gt.size and gt.size == depth.size and depth.size == mask[0].size and mask[0].size == mask[1].size and \
                mask[1].size == mask[2].size and mask[2].size == mask[3].size:#add_revise
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
                masks.append(masks_path)#add
        self.images = images
        self.gts = gts
        self.depths = depths
        self.masks = masks#add

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def binary_masks_loader(self, masks_path):#add
        mask = []#add
        #print('\n\n', len(masks_path), masks_path[0])
        for i in range(4):#add
            img_name = os.path.splitext(os.path.basename(masks_path))[0]#add
            img = Image.open(os.path.join(os.path.dirname(masks_path), img_name + '_ersion_img_' + str(i) + '.png'))#add
            mask.append(img.convert('L'))#add
        return mask#add
    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   depth.resize((w, h),Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


# dataloader for training
#def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):
def get_loader(image_root, gt_root, depth_root, masks_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True):#add_revise
    #dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    dataset = SalObjDataset(image_root, gt_root, depth_root, masks_root, trainsize)#add_revise
    # print(image_root)
    # print(gt_root)
    # print(depth_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.gts[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

