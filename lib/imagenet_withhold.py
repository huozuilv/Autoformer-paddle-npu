import paddle
from PIL import Image
import io


class ImageNet_Withhold(paddle.io.Dataset):

    def __init__(self, data_root, ann_file='', transform=None, train=True,
        task='train'):
        super(ImageNet_Withhold, self).__init__()
        ann_file = ann_file + '/' + 'val_true.txt'
        train_split = task == 'train' or task == 'val'
        self.data_root = data_root + '/' + ('train' if train_split else 'val')
        self.data = []
        self.nb_classes = 0
        folders = {}
        cnt = 0
        self.z = ZipReader()
        f = open(ann_file)
        prefix = 'data/sdb/imagenet' + '/' + ('train' if train_split else 'val'
            ) + '/'
        for line in f:
            tmp = line.strip().split('\t')[0]
            class_pic = tmp.split('/')
            class_tmp = class_pic[0]
            pic = class_pic[1]
            if class_tmp in folders:
                self.data.append((class_tmp + '.zip', prefix + tmp +
                    '.JPEG', folders[class_tmp]))
            else:
                folders[class_tmp] = cnt
                cnt += 1
                self.data.append((class_tmp + '.zip', prefix + tmp +
                    '.JPEG', folders[class_tmp]))
        normalize = paddle.vision.transforms.Normalize(mean=[0.485, 0.456,
            0.406], std=[0.229, 0.224, 0.225])
        if transform is not None:
            self.transforms = transform
        elif train:
            self.transforms = paddle.vision.transforms.Compose([paddle.vision.
                transforms.RandomSizedCrop(224), paddle.vision.transforms.
                RandomHorizontalFlip(), paddle.vision.transforms.ToTensor(),
                normalize])
        else:
            self.transforms = paddle.vision.transforms.Compose([paddle.vision.
                transforms.Scale(256), paddle.vision.transforms.CenterCrop(
                224), paddle.vision.transforms.ToTensor(), normalize])
        self.nb_classes = cnt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iob = self.z.read(self.data_root + '/' + self.data[idx][0], self.
            data[idx][1])
        iob = io.BytesIO(iob)
        img = Image.open(iob).convert('RGB')
        target = self.data[idx][2]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
