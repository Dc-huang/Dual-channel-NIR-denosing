import os
import time
from glob import glob
import numpy as np
import tensorflow as tf
import random
import cv2
from skimage.metrics import structural_similarity as compare_ssim
def dncnn_clean (input, is_training=True, output_channels=1, num_filters=64, block_name='block_clean'):
    with tf.compat.v1.variable_scope(block_name):
        output = tf.compat.v1.layers.conv2d(input, num_filters, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 19+1):
        with tf.compat.v1.variable_scope(block_name+'%d' % layers):
            output = tf.compat.v1.layers.conv2d(output, num_filters, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.compat.v1.layers.batch_normalization(output, training=is_training))
    with tf.compat.v1.variable_scope(block_name+'block17'):
        output = tf.compat.v1.layers.conv2d(output, output_channels, 3, padding='same',use_bias=False)
    return input - output

def dncnn_noise (input, is_training=True, output_channels=1, num_filters=64, block_name='block_noise'):
    with tf.compat.v1.variable_scope(block_name):
        output = tf.compat.v1.layers.conv2d(input, num_filters, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 19+1):
        with tf.compat.v1.variable_scope(block_name+'%d' % layers):
            output = tf.compat.v1.layers.conv2d(output, num_filters, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.compat.v1.layers.batch_normalization(output, training=is_training))
    with tf.compat.v1.variable_scope(block_name+'block17'):
        output = tf.compat.v1.layers.conv2d(output, output_channels, 3, padding='same',use_bias=False)
    return output
def dualDncnn(y_true,y_input, is_training=True, output_channels = 1, num_filters1=64, num_filters2=64):
    clean_out = dncnn_clean(y_input, is_training, output_channels, num_filters1)
    noise_out = dncnn_noise(y_input, is_training, output_channels, num_filters2)
    disc_out = discriminator(clean_out)
    disc_true = discriminator(y_true)
    output = noise_out + clean_out
    # self.noise,self.clean,self.disc,self.output = dualDncnn(self.X, self.is_training)
    return noise_out,clean_out,disc_out,disc_true,output

#判别器函数
def discriminator(input, reuse=None):
    if input.shape[0] is None:
        print("Input is None!")
        return input
    with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
        # 使用卷积层对输入图像进行特征提取
        conv1 = tf.compat.v1.layers.conv2d(input, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

        pool1 = tf.compat.v1.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.compat.v1.layers.conv2d(pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.compat.v1.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

        # 将提取到的特征通过全连接层进行分类
        flatten = tf.keras.layers.Flatten()(pool2)
        # flatten = tf.reshape(pool2, [-1, tf.shape(pool2)[1] * tf.shape(pool2)[2] * tf.shape(pool2)[3]])
        dense = tf.compat.v1.layers.dense(flatten, units=1024, activation=tf.nn.relu)
        logits = tf.compat.v1.layers.dense(dense, units=1)

        # 对判别结果进行sigmoid激活，输出范围为[0, 1],这里先不用
        out = logits

    return out

def discriminator_loss_fn(real_image, generated_image):
    real_label = tf.ones_like(real_image)
    generated_label = tf.zeros_like(generated_image)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(real_label, real_image) + \
           tf.keras.losses.BinaryCrossentropy(from_logits=True)(generated_label, generated_image)
    return loss

def triplet_loss(y_true, clean_out, noise_out, margin=1.0):
    # 计算距离
    dist_neg = tf.reduce_mean(tf.square(y_true - noise_out), axis=[1,2,3])
    dist_pos = tf.reduce_mean(tf.square(y_true - clean_out), axis=[1,2,3])
    # 计算最近的负样本
    dist_neg_closest = tf.reduce_min(dist_neg, axis=0)
    # 计算最远的正样本
    dist_pos_furthest = tf.reduce_max(dist_pos, axis=0)
    # 计算最近的正样本
    # dist_pos_cloest = tf.reduce_min(dist_pos, axis=0)
    # 执行约束
    loss = tf.reduce_mean(tf.maximum(dist_pos_furthest - dist_neg_closest + margin, 0)) + tf.reduce_mean(tf.maximum(dist_pos, 0))
    return loss

def all_loss(y_true,noisy_images,y_pred,noise_images,disc_images,disc_true):
    #con_criterion = contrastive_loss(y_true,y_pred,noise_images)
    criterion =  tf.nn.l2_loss(y_true - y_pred) + tf.nn.l2_loss(noisy_images - y_true - noise_images)
    disc_criterion = discriminator_loss_fn(disc_true,disc_images) + criterion #+ criterion  # discriminator_loss_real(disc_images) +
    con_criterion = triplet_loss(y_true, y_pred, noise_images)  + disc_criterion
    #loss =  tf.nn.l2_loss(y_true - y_pred)  + tf.nn.l2_loss(noisy_images - y_true - noise_images)
    return con_criterion


filepaths = glob('./data/train/original/*.png') #takes all the paths of the png files in the train folder
filepaths = sorted(filepaths)                           #Order the list of files
filepaths_noisy = glob('./data/train/noisy/*.png')
filepaths_noisy = sorted(filepaths_noisy)
ind = list(range(len(filepaths)))

class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        # build model
        #输入清晰图像
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        #输入模糊图像
        self.X = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim])

        # self.noise = dncnn(self.X, is_training=self.is_training)

        # 噪声,干净图片,判别器,重组模糊图像输出
        self.noise,self.clean,self.disc,self.disc_true,self.output = dualDncnn(self.Y,self.X,self.is_training)
        '''
        这里的损失函数要改，注意权重参数y_true,noisy_images,y_pred,noise_images,disc_images,disc_true,output
        '''
        self.loss = (1.0/batch_size)*all_loss(self.Y,self.X,self.clean,self.noise,self.disc,self.disc_true)

        # self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')
        self.dataset = dataset(sess)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eval_files, noisy_files):
        print("[*] Evaluating...")
        psnr_sum = 0
        
        for i in range(10):
            clean_image = cv2.imread(eval_files[i],cv2.IMREAD_GRAYSCALE)
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ..., np.newaxis]
            noisy = cv2.imread(noisy_files[i],cv2.IMREAD_GRAYSCALE)
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ..., np.newaxis]
            
            output_clean_image = self.sess.run(
                [self.clean],feed_dict={self.Y: clean_image,
                           self.X: noisy,
                           self.is_training: False})
            output_clean_image = (output_clean_image - np.min(output_clean_image)) / (
                        np.max(output_clean_image) - np.min(output_clean_image))
            out2 = np.asarray(output_clean_image)
            psnr = psnr_scaled(clean_image, out2[0,0])
            print("img%d PSNR: %.2f" % (i + 1, psnr))
            psnr_sum += psnr

        avg_psnr = psnr_sum / 10

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)


    def train(self, eval_files, noisy_files, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=20):

        numBatch = int(len(filepaths) * 2)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.compat.v1.summary.scalar('loss', self.loss)
        tf.compat.v1.summary.scalar('lr', self.lr)
        # log_dir = './logs'
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        # writer = tf.compat.v1.summary.FileWriter(log_dir, self.sess.graph)
        merged = tf.compat.v1.summary.merge_all()
        clip_all_weights = tf.compat.v1.get_collection("max_norm")        

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_files, noisy_files)  # eval_data value range is 0-255，summary_writer=writer
        for epoch in range(start_epoch, epoch):
            batch_noisy = np.zeros((batch_size,64,64,1),dtype='float32')
            batch_images = np.zeros((batch_size,64,64,1),dtype='float32')
            for batch_id in range(start_step, numBatch):
              try:
                res = self.dataset.get_batch() # If we get an error retrieving a batch of patches we have to reinitialize the dataset
              except KeyboardInterrupt:
                raise
              except:
                self.dataset = dataset(self.sess) # Dataset re init
                res = self.dataset.get_batch()
              if batch_id==0:
                batch_noisy = np.zeros((batch_size,64,64,1),dtype='float32')
                batch_images = np.zeros((batch_size,64,64,1),dtype='float32')
              ind1 = list(range(res.shape[0]//2))
              ind1 = np.multiply(ind1,2)
              for i in range(batch_size):
                random.shuffle(ind1)
                ind2 = random.randint(0,8-1)
                batch_noisy[i] = res[ind1[0],ind2]
                batch_images[i] = res[ind1[0]+1,ind2]
              _, loss ,output_clean_image,original_image= self.sess.run([self.train_op, self.loss,self.clean,self.Y],
                                                 feed_dict={self.Y: batch_images, self.X: batch_noisy, self.lr: lr[epoch],
                                                            self.is_training: True})
              self.sess.run(clip_all_weights)          
              output_clean_image = (output_clean_image - np.min(output_clean_image)) / (np.max(output_clean_image) - np.min(output_clean_image))
              out2 = np.asarray(output_clean_image)
              psnr = psnr_scaled(original_image[0,0], out2[0, 0])
              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, lr: %.6f ,PSNR:%.2f"
                    % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss, lr[epoch],psnr))
              iter_num += 1
              with open('loss_final.txt', 'a') as file:
                  file.write(str(loss)+'\n')
              # writer.add_summary(summary, iter_num)
              
            if np.mod(epoch + 1, eval_every_epoch) == 0: ##Evaluate and save model
                self.evaluate(iter_num, eval_files, noisy_files) # , summary_writer=writer
                self.save(iter_num, ckpt_dir)
        print("[*] Training finished.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.compat.v1.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, eval_files, noisy_files, ckpt_dir, save_dir): #, temporal
        """Test"""
        # init variables
        tf.compat.v1.global_variables_initializer().run()
        assert len(eval_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        psnr1_sum = 0
        ssim_noisy = 0
        ssim_denoised = 0
        nlpd_noisy = 0
        nlpd_denoised = 0
        len_num = len(eval_files)
        for i in range(len(eval_files)):
            clean_image = cv2.imread(eval_files[i],cv2.IMREAD_GRAYSCALE)
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ..., np.newaxis]
            
            noisy = cv2.imread(noisy_files[i],cv2.IMREAD_GRAYSCALE)
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ..., np.newaxis]
          
            output_clean_image = self.sess.run(
                [self.clean],feed_dict={self.Y: clean_image, self.X: noisy,
                                    self.is_training: False})
            output_noise_image = self.sess.run(
                [self.noise],feed_dict={self.Y: clean_image, self.X: noisy,
                                    self.is_training: False})
            output = self.sess.run(
                [self.output],feed_dict={self.Y: clean_image, self.X: noisy,
                                    self.is_training: False})
            output_disc_image = self.sess.run(
                [self.disc],feed_dict={self.Y: clean_image, self.X: noisy,
                                    self.is_training: False})
            output_clean_image = (output_clean_image - np.min(output_clean_image)) / (np.max(output_clean_image) - np.min(output_clean_image))
            output_noise_image = (output_noise_image - np.min(output_noise_image)) / (
                        np.max(output_noise_image) - np.min(output_noise_image))
            output_disc_image = (output_disc_image - np.min(output_disc_image)) / (
                        np.max(output_disc_image) - np.min(output_disc_image))
            out1 = np.asarray(output_noise_image)
            out2 = np.asarray(output_clean_image)
            out3 = np.asarray(output_disc_image)
            out4 = np.asarray(output)

            psnr1 = psnr_scaled(clean_image, noisy)
            clean_image = clean_image[0, :, :, 0]
            noisy = noisy[0, :, :, 0]
            ssim_value_noisy = compare_ssim(clean_image, noisy, full=False, data_range=1.0)
            nlpd_value_noisy = nlpd(clean_image,noisy)
            out2 = out2[0, 0, :, :, 0]
            psnr = psnr_scaled(clean_image, out2)
            ssim_value_denoised = compare_ssim(clean_image, out2, full=False, data_range=1.0)
            nlpd_value_denoised = nlpd(clean_image, out2)
            print("img%d PSNR: %.2f , noisy PSNR: %.2f" % (i + 1, psnr, psnr1))
            print("img%d SSIM: %.2f , noisy SSIM: %.2f" % (i + 1, ssim_value_denoised, ssim_value_noisy))
            print("img%d NLPD: %.2f , noisy NLPD: %.2f" % (i + 1, nlpd_value_denoised, nlpd_value_noisy))

            psnr_sum += psnr
            psnr1_sum += psnr1
            ssim_noisy += ssim_value_noisy
            ssim_denoised += ssim_value_denoised
            nlpd_noisy += nlpd_value_noisy
            nlpd_denoised += nlpd_value_denoised
         #   cv2.imwrite('./data/denoised/%04d.png' % (i), out2 * 255.0)


        avg_psnr = psnr_sum / len_num
        avg_ssim_d = ssim_denoised / len_num
        avg_nlpd_d = nlpd_denoised / len_num
        avg1_psnr = psnr1_sum / len_num
        avg_ssim_n = ssim_noisy / len_num
        avg_nlpd_n = nlpd_noisy / len_num

        print("--- Test ---- Average PSNR %.2f --- Average SSIM %.2f --- Average NLPD %.2f" % (avg_psnr, avg_ssim_d, avg_nlpd_d))
        print("--- Test noisy---- Average PSNR %.2f --- Average SSIM %.2f --- Average NLPD %.2f" % (avg1_psnr, avg_ssim_n, avg_nlpd_n))
    
class dataset(object):
  def __init__(self,sess):
    self.sess = sess
    seed = time.time()
    random.seed(seed)

    random.shuffle(ind)
    
    filenames = list()
    for i in range(len(filepaths)):
        filenames.append(filepaths_noisy[ind[i]])
        filenames.append(filepaths[ind[i]])

    # Parameters
    num_patches = 8   # number of patches to extract from each image
    patch_size = 64                 # size of the patches
    num_parallel_calls = 1          # number of threads
    batch_size = 32                # size of the batch
    get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=patch_size)
    dataset = (
        tf.data.Dataset.from_tensor_slices(filenames)
        .map(im_read, num_parallel_calls=num_parallel_calls)
        .map(get_patches_fn, num_parallel_calls=num_parallel_calls)
        .batch(batch_size)
        .prefetch(batch_size)
    )
    
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    self.iter = iterator.get_next()
  

  def get_batch(self):
        res = self.sess.run(self.iter)
        return res
        
def im_read(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image
    
def get_patches(image, num_patches=128, patch_size=64):
    patches = []
    for i in range(num_patches):
      point1 = random.randint(0,116) # 116 comes from the image source size (180) - the patch dimension (64)
      point2 = random.randint(0,116)
      patch = tf.image.crop_to_bounding_box(image, point1, point2, patch_size, patch_size)
      patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 1]
    return patches
    
def cal_psnr(im1, im2): # PSNR function for 0-255 values
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr
    
def psnr_scaled(im1, im2): # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 **2 / mse)
    return psnr

"""计算两个图像的 Laplacian 金字塔"""
def laplacian_pyramid(img, levels):
    G = img.copy()
    gp = [G]
    for i in range(levels):
        G = cv2.pyrDown(G)
        gp.append(G)
    lp = [gp[levels]]
    for i in range(levels, 0, -1):
        GE = cv2.pyrUp(gp[i])
        L = cv2.subtract(gp[i - 1], GE)
        lp.append(L)
    return lp

"""计算 NLPD"""
def nlpd(img1, img2, levels=3):
    # 将图像转换为双精度型
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    # 计算 Laplacian 金字塔
    img1_lp = laplacian_pyramid(img1, levels)
    img2_lp = laplacian_pyramid(img2, levels)

    # 计算每个金字塔层的标准差
    stds = np.zeros((levels + 1, 1))
    for i in range(levels + 1):
        stds[i] = np.std(img1_lp[i])

    # 计算每个金字塔层上的 NLPD
    nlpd_vals = np.zeros((levels + 1, 1))
    for i in range(levels + 1):
        diff = np.subtract(img1_lp[i], img2_lp[i])
        diff_norm = np.linalg.norm(diff, 'fro')
        nlpd_vals[i] = 1.0 / (img1_lp[i].shape[0] * img1_lp[i].shape[1]) * diff_norm / (stds[i] + np.finfo(float).eps)

    # 将所有金字塔层上的 NLPD 求和得到最终的 NLPD
    nlpd_val = np.sum(nlpd_vals) / (levels + 1)
    return nlpd_val*100