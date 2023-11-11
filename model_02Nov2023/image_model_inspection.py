import tensorflow_datasets as tfds
import tensorflow as tf
import holoviews as hv
import numpy as np
import pandas as pd
from functools import cache
from typing import Literal

from plotutils import histo
import string, random

def random_axis(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def vis_one_example(x):
    img = hv.RGB(x.numpy())
    return img.opts(frame_width=350, frame_height=350)

def hv_class_prob(y_pred, all_labels, yaxis='prob', xaxis='class', true_label_index = 0):
    all_bar = hv.Bars((all_labels, y_pred), xaxis, yaxis).opts(xrotation=45, width=600).sort(yaxis, reverse=True)
    if not true_label_index is None: 
        highlighted_bar = hv.Bars(([all_labels[true_label_index]], [y_pred[true_label_index]]), xaxis, yaxis).opts(color='red')
        return all_bar*highlighted_bar
    return all_bar

def hv_confusion(confusion_matrix, labels=[], kdims=['Prediction', 'Real']):
    confusion_matrix = confusion_matrix / confusion_matrix.sum()  
    axes = np.arange(confusion_matrix.shape[0]) if not len(labels) else labels
    
    return hv.HeatMap((axes, axes, confusion_matrix), kdims=kdims).opts(colorbar=True, frame_width=350, frame_height=350, xrotation=90)

def join_images(image, max_column=6, filler=np.nan):
    """    
    it unstack the images and tile them next to each other 

    image: a stack of images in an ndarray with a shape (x, y, ..., n), where n is the number of images

    """
    n_image = image.shape[-1]

    # the fillers 
    num_filler_needed = max_column - (n_image % max_column)
    filler_img = np.zeros_like(image[..., 0:1])
    filler_img[:] = filler
    filler_img = np.tile(filler_img, reps = num_filler_needed)
    


    # put in the fillers
    image = np.concatenate([image, filler_img], axis=-1)
    image_idx = 0
    rows = []

    
    while True:
        next_img_idx = image_idx+max_column

        row_ = np.concatenate([image[..., i] for i in range(image_idx, next_img_idx)], axis=1)
        rows.append(row_)
        if next_img_idx >= n_image:
            break
        image_idx = next_img_idx


    return np.concatenate(rows, axis=0)


def break_down_RGB(image):

    R = image.copy()
    R[..., 1:] = 0

    G = image.copy()
    G[..., 0] = 0    
    G[..., 2] = 0    
    
    B = image.copy()
    B[..., :-1] = 0

    return np.concatenate((R, G, B), axis=1)


def show_filters(filter_images, max_column = 6, scale=1.5, sep_line_color='red'):
    n_filter = filter_images.shape[-1]
    

    n_col = min(n_filter, max_column) 
    n_row = int(np.ceil(n_filter / max_column))

    joint_image = join_images(filter_images)

    xaxis = np.arange(joint_image.shape[1])*scale
    yaxis = np.arange(joint_image.shape[0])*scale


    hlines = hv.Overlay([hv.HLine((-h-1)*filter_images.shape[1]*scale+scale/2).opts(color=sep_line_color, line_width=0.5) for h in range(-1, n_row)])
    vlines = hv.Overlay([hv.VLine((v+1)*filter_images.shape[0]*scale-scale/2).opts(color=sep_line_color, line_width=0.5) for v in range(-1, n_col)])

    return hv.Image((xaxis, -yaxis, 1-joint_image), kdims=[random_axis(), random_axis()]).opts(
        aspect='equal', cmap='greys'
    ).opts(xaxis=None, yaxis=None)*hlines*vlines



class ImageModelInspector:
    def __init__(self, model, train_ds, test_ds, labels, batched=False):
        """
        the dataset shouldn't be shuffled nor batched
        """
        self.model = model 
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.labels = labels

        assert not batched

    @staticmethod
    def from_cifar100_coarse(model, batched=False):

        # data prep 
        ds, ds_info = tfds.load('cifar100', with_info=True)

        ds_train = ds['train'].prefetch(tf.data.AUTOTUNE)
        ds_train = ds_train.map(lambda x: (x['image']/255, x['coarse_label']))

        ds_test = ds['test'].prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.map(lambda x: (x['image']/255, x['coarse_label']))

        labels = list(map(ds_info.features['coarse_label'].int2str, range(20)))
        return ImageModelInspector(model, ds_train, ds_test, labels, batched=batched)
    
    def show_predictions(
            self, 
            from_ds: Literal['test', 'train'] = 'test', 
            img_pixsize = (400, 400), 

        ):
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        ds_iter = iter(use_ds)
        # TODO: no need to do iter?
        for x, y_real in ds_iter:
            
            y_pred_prob = self.model(x[None, ...])

            yield (vis_one_example(x) + hv_class_prob(y_pred_prob[0], self.labels, true_label_index=y_real))

    @cache
    def y_pred_prob(self, from_ds: Literal['test', 'train'] = 'test'):
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        return self.model.predict(use_ds.batch(128))
    
    @cache
    def y_pred(self, from_ds: Literal['test', 'train'] = 'test'):
        return tf.argmax(self.y_pred_prob(from_ds), axis=1)
    
    def y_true(self, from_ds: Literal['test', 'train'] = 'test'):
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        
        return np.array(list(use_ds.map(lambda x, y: y).as_numpy_iterator()))
        


    @cache
    def confusion_matrix(
            self, 
            from_ds: Literal['test', 'train'] = 'test'
        ):       

        y_pred = self.y_pred(from_ds)
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        y_real = np.array(list(use_ds.map(lambda x, y: y).as_numpy_iterator()))
        
        return tf.math.confusion_matrix(y_real, y_pred).numpy()
    
    def show_confusion_matrix(
            self, 
            from_ds: Literal['test', 'train'] = 'test', 
            mask_eye = True
        ):
        mask = ~(np.eye(len(self.labels)).astype(bool))
        return  hv_confusion(self.confusion_matrix(from_ds)*mask, self.labels)


    def top_mismatches_by_category(self, from_ds: Literal['test', 'train'] = 'test'):

        mask = ~(np.eye(len(self.labels)).astype(bool))

        confusion_matrix_df = pd.DataFrame(self.confusion_matrix(from_ds), index=self.labels, columns = self.labels)

        return (confusion_matrix_df*mask).idxmax(1)
    
    
    def top_mismatches(self, from_ds: Literal['test', 'train'] = 'test'):

        mask = ~(np.eye(len(self.labels)).astype(bool))

        confusion_matrix_df = pd.DataFrame(self.confusion_matrix(from_ds), index=self.labels, columns = self.labels)

        return (confusion_matrix_df*mask).stack().sort_values(ascending=False).head(20)
    

    def classification_count(self, from_ds: Literal['test', 'train'] = 'test', normalise=True):

        counts = pd.Series(list(map(lambda x: self.labels[x], self.y_pred(from_ds)))).value_counts()
        if normalise:
            counts = counts/counts.sum(0)
        return counts
    
    def loss_by_category(self, from_ds: Literal['test', 'train'] = 'test'):
        y_true = self.y_true(from_ds)
        y_pred_prob = self.y_pred_prob(from_ds)
        
        loss_byC = {}
        for c_ in range(len(np.unique(y_true))):
            bIdx = (y_true == c_)
            loss_byC[self.labels[c_]] = self.model.loss(y_true[bIdx], y_pred_prob[bIdx]).numpy()
        return pd.Series(loss_byC, name='loss').sort_values(ascending=False)
    
    def get_filter_layers(self):
        return [l for l in self.model.layers if len(l.output.shape) > 2]

    def get_dense_layers(self):
        return [l for l in self.model.layers if 'dense.Dense' in str(type(l))]
    

    def dense_layer_output_summary(self, nth_dense_layer, from_ds : Literal['test', 'train'] = 'test', ):
        """
        medians are all zero. 

        sparse! 

        because of relu. How about the distribution of the non-zeros? Any outliners?
        """
        layer = self.get_dense_layers()[nth_dense_layer]
        
        _model = tf.keras.models.Model(self.model.input, layer.output) 
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds

        outputs = _model.predict(use_ds.batch(128))
        
        return abs(outputs).mean(0), outputs.mean(0), outputs.std(0), np.median(outputs, axis=0)
    

    def dense_layer_weight_bias_dist(self, nth_dense_layer):
        """
        try leaky relu
        """

        layer = self.get_dense_layers()[nth_dense_layer]

        weight, bias = layer.weights[0].numpy(), layer.weights[1].numpy()


        return histo(bias, 50, ['bias', 'freq'])  + histo(weight.std(0), 50,  ['weight.std(0)', 'freq']) + histo(abs(weight).mean(0), 50,  ['abs(weight).mean(0)', 'freq'])
    

    def show_image(
            self, from_ds : Literal['test', 'train'] = 'test', 
        ):
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        ds_iter = iter(use_ds)
        for x, _ in ds_iter:
            img = x.numpy()
            yield hv.RGB((np.arange(img.shape[1]), -np.arange(img.shape[0]), img), kdims=[random_axis(), random_axis()]).opts(aspect='equal', xaxis=None, yaxis=None)

    def show_RGB(
            self, from_ds : Literal['test', 'train'] = 'test', 
        ):
        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        ds_iter = iter(use_ds)

        for x, _ in ds_iter:
            img = break_down_RGB(x.numpy())
            yield hv.RGB((np.arange(img.shape[1]), -np.arange(img.shape[0]), img), kdims=[random_axis(), random_axis()]).opts(aspect='equal', xaxis=None, yaxis=None)

    def show_filter_images(
            self, nth_filter_layer, 
            from_ds : Literal['test', 'train'] = 'test', 
            
        ):
        layer = self.get_filter_layers()[nth_filter_layer]

        _model = tf.keras.models.Model(self.model.input, layer.output) 

        use_ds = self.test_ds if not from_ds == 'train' else self.train_ds
        ds_iter = iter(use_ds)

        for x, _ in ds_iter:
            
            filter_image = _model(x[None, ...])
            
            
            yield show_filters(filter_image.numpy()[0])

    def show_all_images(self):
        origin_gen = self.show_image()
        img_rgb_gen = self.show_RGB()
        filter_gens = [self.show_filter_images(i) for i in range(len(self.get_filter_layers()))]
        all_gens = [origin_gen, img_rgb_gen] + filter_gens

        while True:
            yield hv.Layout([next(gen_) for gen_ in all_gens]).cols(1)
