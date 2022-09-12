import numpy as np
import torch
from   torchmetrics import StructuralSimilarityIndexMeasure
import torchvision.transforms as transforms
from   skimage.transform import resize, rescale
import torch.nn  as nn
import cv2
import glob
from   tqdm import tqdm
from   convolution_net import UNet, Convolution_NxN, Convolution_3
from   customDatset import Custom_Data
from   torch.utils.data import DataLoader


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#----------------------------loding Data--------------------------#
def test_loader(path):
    file      = sorted(glob.glob(path))
    data_list = []
    for image in file:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        #image = cv2.resize(image, (256,256))
        data_list.append(image)
    return data_list

def load_data(truth_d, deblured_d):
    data_set      = Custom_Data(truth_d, deblured_d
                          ) 
    data_loader = DataLoader(dataset= data_set ,batch_size=10, shuffle=True)
    return data_loader
#---------------------------loss function------------------------#
loss_func = nn.MSELoss() 
def custom_loss_function(a, pred, truth):
    l1    = nn.L1Loss()
    ssim  = StructuralSimilarityIndexMeasure()
    l_mix = a*l1(pred, truth)+(1-a)*ssim(pred,truth)
    return l_mix
#---------------------------Training------------------------------#

def training(epochs, traindata):
    for epoch in tqdm(range(epochs)):
        for i, sample in enumerate(traindata):
            x      = sample['input'].to(device) 
            y      = sample['target'].to(device)
            y      = y.float() 
            x      = x.float()
            output = model(x)
            loss   = loss_func(output, y)
            loss.backward()                                                              
            optimiser.step() 
            optimiser.zero_grad() 
            output_cpu = output.to('cpu')
            cnn_output = np.squeeze(output_cpu.detach().numpy())
            if epoch % 10== 0 : # printig evry ten step 
                print( f'epoch {epoch+1}: loss = {loss:.8f}')
    cv2.waitKey(0)

    print('Training is done !!')

#---------------------Testing the model-----------#
def testing(test_data):
    with torch.no_grad():
        for i,image in enumerate(test_data):
            image      = torch.from_numpy(image /255).float().unsqueeze(0).unsqueeze(0)
            image      = image.to(device)
            output     = model(image)
            output_cpu = output.to('cpu')
            cnn_output = np.squeeze(output_cpu.detach().numpy())
            cv2.imwrite(outpath + f'{str(i)}.png', (cnn_output*255).astype(np.uint8))

#----------------------- main --------------#

if __name__ == '__main__':
    outpath               = 'C:/Users/Morhaf.Haedar/Desktop/Andreas/Viscom/cnn_output/xray/Prep' 
    ground_truth_dir      = 'C:/Users/Morhaf.Haedar/Desktop/Mariano/Run240_Slow/*.png'
    filtered_data_dir     = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Andreas/training/Filtred_data/method5(27,2)/Train_set/*.png'
    test_data_dir         = "C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Andreas/training/Filtred_data/method5(27,2)/Test_set/*.*"
    epochs                = 550
    training_data         = load_data(ground_truth_dir, filtered_data_dir)
    model                 = Convolution_3(5).to(device)
    loss_func             = nn.MSELoss()
    optimiser             = torch.optim.Adam(model.parameters(), lr = 0.007)
    training(epochs, training_data)
    file = 'model7.pth'
    torch.save(model.state_dict(), file)
    test_data = test_loader(test_data_dir)
    testing(test_data)


'''
pathes on my pc:

    outpath               = 'C:/Users/Morhaf.Haedar/Desktop/Andreas/Viscom/cnn_output/filter5/ima' 
    ground_truth_dir      = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Andreas/Sliced_with_An_code/Gray/Train_set/*.png'
    filtered_data_dir     = 'C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Andreas/training/Filtred_data/method5(27,2)/Train_set/*.png'
    test_data_dir         = "C:/Users/morha/OneDrive - stud.uni-hannover.de/Desktop/Viscom/Andreas/training/Filtred_data/method5(27,2)/Test_set/*.*"
'''