import wandb
import numpy as np

DATASET_MEAN = 239.74347588572797 / 4096
DATASET_STD = 397.1364638124688 / 4096

def tensor2im_dicom(input_image, label, imtype=np.uint8):
    """"Converts a Tensor array, which is a dicom image into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if label == 'real_image':
        image_numpy = input_image.data.cpu().float().numpy()[0, :, :]
        input_image = np.expand_dims(input_image[0, :, :], 0)
        #image_numpy = image_numpy * DATASET_STD + DATASET_MEAN
        image_numpy = np.tile(image_numpy, (3,1,1)).transpose(1,2,0)
        #image_numpy = image_numpy / np.max(image_numpy) * 255                                              
        #image_numpy = cv2.equalizeHist(image_numpy)
        #image_numpy = np.tile(image_numpy, (1,1,3))

        image_numpy = (image_numpy - np.min(image_numpy.flatten())) / (np.max(image_numpy.flatten()) - np.min(image_numpy.flatten()))
        image_numpy = image_numpy * 255
    elif label == 'sample':
        image_numpy = input_image.data.cpu().float().numpy()[0, :, :]
        # print min and max values of the image
        #print('min = %3.3f, max = %3.3f' % (np.min(image_numpy), np.max(image_numpy)))
        #image_numpy = (image_numpy + 1) / 2
        image_numpy = image_numpy * DATASET_STD + DATASET_MEAN
        image_numpy = np.tile(image_numpy, (1,1,1)).transpose(1,2,0) 
        #image_numpy = (image_numpy + 1) / 2 # generated image is in [-1, 1]
        #image_numpy = image_numpy / np.max(image_numpy) * 255
        image_numpy = (image_numpy - np.min(image_numpy.flatten())) / (np.max(image_numpy.flatten()) - np.min(image_numpy.flatten()))
        image_numpy = image_numpy * 255   
        
    elif label == 'mask':
        input_image = np.expand_dims(input_image[0, :, :], 0)
        image_numpy = np.tile(input_image, (3,1,1)).transpose(1,2,0) * 255

    image_numpy = image_numpy.astype(imtype)  

    return image_numpy

class Visualizer():

    def __init__(self, wandb_run=None, image_log_interval=0) -> None:
        self.wandb_run = wandb_run
        self.visuals = {}

    def save_images():
        pass
    


    def log_images(self, sample, batch, cond, step):
        if self.wandb_run is not None:
            columns = ['mask', 'sample', 'real_image']
            columns.insert(0, 'step')
            #result_table = wandb.Table(columns=columns)
            table_row = [step]
            ims_dict = {}
            
            image_numpy = tensor2im_dicom(sample, 'sample')                
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict['sample'] = wandb_image

            image_numpy = tensor2im_dicom(cond['label_ori'].float(), 'mask')                
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict['mask'] = wandb_image

            image_numpy = tensor2im_dicom(batch, 'real_image')                
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict['real_image'] = wandb_image


            self.wandb_run.log(ims_dict)
    

            
