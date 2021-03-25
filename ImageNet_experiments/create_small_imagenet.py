import os
import shutil
import numpy as np
root = '/mnt/ramdisk/'


def sample_per_class(src_root, target_root, n=10):
    target_root = os.path.join(target_root, 'small_imagenet_class_1000_n_{}'.format(n), 'train')
    for category in os.listdir(src_root):
        category_root = os.path.join(src_root, category)
        filenames = os.listdir(category_root)

        selected_files = np.random.choice(len(filenames), n, replace=False)
        dirname = os.path.join(target_root, category)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for i in selected_files:
            shutil.copy(os.path.join(category_root, filenames[i]), os.path.join(dirname, filenames[i]))


#sample_per_class('/mnt/ramdisk/ImageNet/train', '/opt/caoyh/datasets/small_imagenet', n=50)

def sample_class(src_root, target_root, class_number=10, total_number=10000):
    target_root = os.path.join(target_root, 'small_imagenet_class_{}_total_{}'.format(class_number, total_number), 'train')

    categories = os.listdir(src_root)
    selected_categories = np.random.choice(len(categories), class_number, replace=False)

    n = total_number // class_number

    for i in selected_categories:
        category = categories[i]

        category_root = os.path.join(src_root, category)
        filenames = os.listdir(category_root)

        selected_files = np.random.choice(len(filenames), n, replace=False)
        dirname = os.path.join(target_root, category)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for i in selected_files:
            shutil.copy(os.path.join(category_root, filenames[i]), os.path.join(dirname, filenames[i]))

#sample_class('/mnt/ramdisk/ImageNet/train', '/opt/caoyh/datasets/small_imagenet')


def sample_total(src_root, target_root, total_number=10000):
    target_root = os.path.join(target_root, 'small_imagenet_total_{}'.format(total_number), 'train')

    all_filenames = []
    for category in os.listdir(src_root):
        category_root = os.path.join(src_root, category)
        filenames = os.listdir(category_root)
        for filename in filenames:
            temp = '{}/{}'.format(category, filename)
            all_filenames.append(temp)

    print(len(all_filenames))

    selected_files = np.random.choice(len(all_filenames), total_number, replace=False)

    for i in selected_files:
        filename = all_filenames[i]
        category, name = filename.split('/')[0], filename.split('/')[1]
        category_root = os.path.join(src_root, category)

        dirname = os.path.join(target_root, category)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        shutil.copy(os.path.join(category_root, name), os.path.join(dirname, name))

sample_total('/mnt/ramdisk/ImageNet/train', '/opt/caoyh/datasets/small_imagenet', total_number=50000)
