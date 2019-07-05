#%%
from classes.Generator import Generator as G
from classes.Discriminator import Discriminator as D
from classes.Dataset import ImageDataset as TSet
from torchvision import transforms as t
from torch.nn import BCELoss as bce
from torch.optim import Adam
from random import uniform as uni
from torch import Tensor, from_numpy as numpyToTensor, load, save
from numpy.random import rand as randomNpArray
from matplotlib import pyplot as plt, animation as anim
from math import floor

#%%

# SETTINGS #
training_mode = True
load_weights = True
epochs = 1000000

#%%
if(not training_mode):
    epochs = 1
#%%
generator = G()
discriminator = D()

generator.cuda()
discriminator.cuda()

#%%
if(load_weights):
    generator.load_state_dict(load('./generator.pth'))
    generator.eval()
    discriminator.load_state_dict(load('./discriminator.pth'))
    discriminator.eval()

#%%
if(training_mode):
    generator.train()
    discriminator.train()

#%%
criterion_d = bce()
criterion_g = bce()

#%%
optimizer_d = Adam(discriminator.parameters(), lr=1e-7)
optimizer_g = Adam(generator.parameters(), lr=1e-5)
#%%
if(training_mode):
    transform = t.Compose([
        t.Resize((89, 89)),
        t.ToTensor()
    ])

#%%
if(training_mode):
    training_set = TSet(transform_in=transform)

#%%
if(training_mode):
    realLossessAveragesArray = []
    fakeLossessAveragesArray = []
    generatorLossessAveragesArray = []
    noise1Array = []
    noise2Array = []
    tensorReal = Tensor([1]).cuda()
    tensorFake = Tensor([0]).cuda()
    for image in range(len(training_set)):
        npArray = randomNpArray(1, 1, 20, 20)
        randomTensor = numpyToTensor(npArray).cuda().float()    
        noise1Array.append(randomTensor)

        npArray = randomNpArray(1, 1, 20, 20)
        randomTensor = numpyToTensor(npArray).cuda().float()    
        noise2Array.append(randomTensor)

#%%
generator(randomTensor).size()

#%% 
for epoch in range(epochs):
    if(training_mode):
        realLossesArray = []
        fakeLossesArray = []
        generatorLossesArray = []
        for imageIndex, image in enumerate(training_set):

            guess_real = discriminator(image)
            loss_d_real = criterion_d(guess_real, tensorReal )
            realLossesArray.append(loss_d_real.item())

            randomTensor = noise1Array[imageIndex]
            generatedImage = generator(randomTensor)  

            guess_fake = discriminator(generatedImage)
            loss_d_fake = criterion_d(guess_fake, tensorFake )
            fakeLossesArray.append(loss_d_fake.item())        

            optimizer_d.zero_grad()
            loss_d = loss_d_fake + loss_d_real
            loss_d.backward()
            optimizer_d.step()

            # -------------------

            randomTensor = noise2Array[imageIndex]    
            generatedImage = generator(randomTensor)

            optimizer_g.zero_grad()

            loss_g = criterion_g( discriminator(generatedImage), tensorReal )
            generatorLossesArray.append(loss_g.item())
            
            loss_g.backward()
            optimizer_g.step()

        realLossessAveragesArray.append(sum(realLossesArray) / len(realLossesArray))
        fakeLossessAveragesArray.append(sum(fakeLossesArray) / len(fakeLossesArray))
        generatorLossessAveragesArray.append(sum(generatorLossesArray) / len(generatorLossesArray))
    else:
        npArray = randomNpArray(1, 3, 89, 89)
        randomTensor = numpyToTensor(npArray).cuda().float()    
        generatedImage = generator(randomTensor)            

    if(training_mode):
        if(epoch % 100 == 0):
            print("Epoch: ", epoch)
            fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
            fig.set_figwidth(5)
            fig.set_figheight(5)    
            plotImage = t.ToPILImage()(generatedImage[0].cpu().detach())
            axs.imshow(plotImage)
            plt.pause(0.001)
    

#%%
if(training_mode):
    save(generator.state_dict(), './generator.pth')
    save(discriminator.state_dict(), './discriminator.pth')

#%%
if(training_mode):
    plt.plot(realLossessAveragesArray, color='g')
    plt.plot(fakeLossessAveragesArray, color='r')
    plt.plot(generatorLossessAveragesArray, color='w')
    plt.ylim(0.0, 5)
    plt.show()

#%%
