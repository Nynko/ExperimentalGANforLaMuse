
from ast import Lambda
import os
from sys import displayhook
import time
import matplotlib.pyplot as plt
import tensorflow as tf

channels = 3
gf = 64
df = 64
img_shape = (1024,1024,channels)


  
# =================================================================================== #
#               8. Define the loading function                                        #
# =================================================================================== #  
def get_batch(imgs_index, batch_imgs):
    if(imgs_index+batch_imgs) >= len(imgs_in_path):
        batch_imgs = len(imgs_in_path)-imgs_index
    real_imgs = np.zeros((batch_imgs, img_width, img_height,3))
    masks = np.zeros((batch_imgs, img_width, img_height,1))
    masked_imgs = np.zeros((batch_imgs, img_width, img_height,3))
    masks_index = random.sample(range(1,len(masks_in_path)), batch_imgs)
    maskindx = 0
    for i in range(batch_imgs):
        print("\rLoading image number "+ str(i) + " of " + str(len(imgs_in_path)), end = " ")
        real_img = cv2.imread(img_dir + imgs_in_path[imgs_index], 1).astype('float')/ 127.5 -1
        real_img = cv2.resize(real_img,(img_width, img_height))
        #If masks bits are white, DO NOT subtract from 1.
        #If masks bits are black, subtract from 1.
        mask = 1-cv2.imread(masks_dir + masks_in_path[masks_index[maskindx]],0).astype('float')/ 255
        mask = cv2.resize(mask,(img_width, img_height))
        mask = np.reshape(mask,(img_width, img_height,1))

        masks[i] = mask
        real_imgs[i] = real_img
        #masked_imgs[np.where((mask ==[1,1,1]).all(axis=2))]=[255,255,255]
        masked_imgs[i][np.where(mask == 0)]=1
        maskindx +=1
        imgs_index +=1
        if(imgs_index >= len(imgs_in_path)):
            imgs_index = 0
#            cv2.imwrite(os.path.join(path, 'mask_'+str(i)+'.jpg'),rawmask)
#            cv2.imshow("mask",((masked_imgs[0]+1)* 127.5).astype("uint8"))
#            cv2.waitKey(0 )
    return imgs_index,real_imgs, masks,masked_imgs
     
# =================================================================================== #
#               8. Define the loading function                                        #
# =================================================================================== #
def train(self):
    # Ground truths for adversarial loss
    valid = np.ones([batch_size, 1])
    fake = np.zeros((batch_size, 1))
    total_files= 27000
    batch_imgs = 1000
    imgs_index =0
    dataLoads = total_files//batch_imgs
    generator.load_weights(r'./{}/{}/weight_{}.h5'.format(models_path, dataset_name, last_trained_epoch))
    print ( "Successfully loaded last check point" )
    for epoch in range(1, num_epochs + 1):
        for databatchs in range(dataLoads):
            imgs_index,imgs, masks,masked_imgs = get_batch(imgs_index, batch_imgs)
            batches = imgs.shape[0]//batch_size
            global_step = 0
            for batch in range(batches):
                idx = np.random.permutation(imgs.shape[0])
                idx_batches = idx[batch*batch_size:(batch+1)*batch_size]
                gen_imgs=generator.predict([imgs[idx_batches],masks[idx_batches]], batch_size)
                gen_imgs = gen_imgs[:,:,:,0:3]

                # =================================================================================== #
                #                             8.2. Train the discriminator                            #
                # =================================================================================== # 
                d_loss_real = discriminator.train_on_batch(imgs[idx_batches], valid)
                d_loss_fake = discriminator.train_on_batch(gen_imgs[:,:,:,0:3], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


                # =================================================================================== #
                #                             8.3. Train the generator                                #
                # =================================================================================== #

                # Train the generator
                g_loss = combined.train_on_batch([imgs[idx_batches] ,masks[idx_batches]], [imgs[idx_batches],valid])

                # =================================================================================== #
                #                             8.4. Plot the progress                                  #
                # =================================================================================== #
                print ("Epoch: %d Batch: %d/%d dataloads: %d/%d [D loss: %f, op_acc: %.2f%%]  [G loss: %f MSE loss: %f]" % (epoch+current_epoch,
                            batch, batches,databatchs,dataLoads, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))


        idx_batches = idx[databatchs*batch_size:(databatchs+1)*batch_size]
        imgs = imgs[idx]
        masks = masks[idx]

        input_img = np.expand_dims(imgs[0], 0)
        input_mask = np.expand_dims(masks[0], 0)

        if epoch % 1 == 0:
            if not os.path.exists("{}/{}/".format(models_path, dataset_name)):
                    os.makedirs("{}/{}/".format(models_path, dataset_name))
            name = "{}/{}/weight_{}.h5".format(models_path, dataset_name, epoch+current_epoch)

            generator.save_weights(name)
            if not os.path.exists(dataset_name):
                os.makedirs(dataset_name,exist_ok=True)
            predicted_img = generator.predict([input_img, input_mask])
            sample_images(dataset_name, input_img, predicted_img[:,:,:,0:3],
                                    input_mask, epoch)
    print("Total Processing time:: {:4.2f}min" .format((end_time - start_time)/60))
    epoch+=1

# =================================================================================== #
#               9. Sample images during training                                      #
# =================================================================================== # 
    
def sample_images(self, dataset_name,input_img, sample_pred, mask, epoch):
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    input_img = np.expand_dims(input_img[0], 0)
    input_mask = np.expand_dims(mask[0], 0)
    maskedImg = ((1 - input_mask)*input_img) + input_mask       
    img = np.concatenate((((maskedImg[0]+1)* 127.5).astype("uint8"),
                            ((sample_pred[0]+1)* 127.5).astype("uint8"),
                            ((input_img[0]+1)* 127.5).astype("uint8")),axis=1)
    img_filepath = os.path.join(dataset_name, 'pred_{}.jpg'.format(epoch+current_epoch))

    cv2.imwrite(img_filepath, img) 

# =================================================================================== #
#               10. Plot the discriminator and generator losses                       #
# =================================================================================== # 
    
def plot_logs(self,epoch, avg_d_loss, avg_g_loss):
    if not os.path.exists("LogsUnet"):
        os.makedirs("LogsUnet")
    plt.figure()
    plt.plot(range(len(avg_d_loss)), avg_d_loss,
                color='red', label='Discriminator loss')
    plt.plot(range(len(avg_g_loss)), avg_g_loss,
                color='blue', label='Adversarial loss')
    plt.title('Discriminator and Adversarial loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Adversarial/Discriminator)')
    plt.legend()
    plt.savefig("LogsUnet/{}_paper/log_ep{}.pdf".format(dataset_name, epoch+current_epoch))


