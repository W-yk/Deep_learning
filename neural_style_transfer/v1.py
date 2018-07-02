from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b

base_image_path = "D:/Git/Deep_Learning/neural_style_transfer/img/content4.jpg"
style_reference_image_path = "D:/Git/Deep_Learning/neural_style_transfer/img/style.jpg"

# scale the imput image into a trainable size
width, height = load_img(base_image_path).size                  
img_nrows = 400                                                 
img_ncols = int(width * img_nrows / height)     

content_weight=1
style_weight=1000

#process the image for vgg19 model to accept
def preprocess_image(image_path):
    img = load_img(image_path,target_size=(img_nrows,img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

#restore the image by doing the opposite of 'preprocess_image'
def deprocess_image(x):
    x = np.reshape(x,[img_nrows, img_ncols, 3])
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(x):
    assert K.ndim(x) == 3
    #change the input dimension into [n_C,n_H,n_W]
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    #compute gram matrix with dimension [n_C, n_C]
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return K.sum(K.square(S - G)) 
                                     
def content_loss(content, generated):
    return K.sum(K.square(generated - content))   



# packaging the cost funtion
content = K.variable(preprocess_image(base_image_path))
style = K.variable(preprocess_image(style_reference_image_path))
generated = K.placeholder((1, img_nrows, img_ncols, 3),name='x')
input_tensor = K.concatenate([content,style,generated], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, generated)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([generated], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)

for i in range(21):
    print('Start of iteration', i)
    
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    if i%4==0:
        img = deprocess_image(x.copy())
        fname = "D:/Git/Deep_Learning/neural_style_transfer/output" + '_at_iteration_%d1.png' % i
        save_img(fname, img)
        print('Image saved as', fname)