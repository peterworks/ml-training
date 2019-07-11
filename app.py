#%%
from classes.Generator import Generator as G
from classes.Discriminator import Discriminator as D
from classes.Dataset import ImageDataset as TSet
from classes.Plotter import Plotter
from classes.NoiseGenerator import NoiseGenerator as NGen
from torchvision import transforms as t
from torch.nn import BCELoss as bce, L1Loss as l1, MSELoss as mse
from torch.optim import Adam
from random import uniform as uni
from torch import Tensor, from_numpy as numpyToTensor, load, save
from numpy.random import normal as randomNpArray
from numpy import float32 as f32, ones, zeros, full
from torch.utils.data import DataLoader as dl
from matplotlib import pyplot as plt

#%%

# SETTINGS #
training_mode = True
load_weights = False
epochs = 50000
lookup = False
batchSize = 16
channels = 3
noiseWidth = 24
noiseHeight = 24
imgDim = (46, 46)
learningRate = 2e-4
epochsPerPrint = 10
generatorIterationsRatio = 10
discriminatorIterationsRatio = 1
dataRoot = './Data'

#%%
if(lookup):
    batchSize = 9
    training_mode = False
    dataRoot = './Lookup'

#%%
if(not training_mode):
    epochs = 1
#%%
generator = G(lookup=lookup)
discriminator = D(lookup=lookup, batch_size=batchSize)
plotter = Plotter(batchSize)
noiseGenerator = NGen()

generator.cuda()
discriminator.cuda()

#%%
if(load_weights):
    generator.load_state_dict(load('./generator.pth'))
    discriminator.load_state_dict(load('./discriminator.pth'))

#%%
if(load_weights):
    generator.eval()
    discriminator.eval()

#%%
if((not lookup) and training_mode):
    generator.train()
    discriminator.train()

#%%
criterion_d = bce()
criterion_g = bce()

criterion_d.cuda()
criterion_g.cuda()

#%%
optimizer_d = Adam(discriminator.parameters(), lr=learningRate)
optimizer_g = Adam(generator.parameters(), lr=learningRate)
#%%
if(training_mode or lookup):
    transform = t.Compose([
        t.Resize(imgDim),
        t.ToTensor(),
        t.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])

#%%
if(training_mode or lookup):
    training_set = TSet(root=dataRoot, transform_in=transform)

#%%
if(training_mode or lookup):
    training_loader = dl(training_set, batch_size=batchSize, shuffle=True, drop_last=True)

#%%

if(lookup):
    for imageIndex, image in enumerate(training_loader, 0):
        if(imageIndex > 2):
            break
        image = discriminator(image.cuda())  
        plotter.plotImage(image[0])
        plotter.plotImage(image[1])

    noise = noiseGenerator.generate(batchSize, channels, (noiseHeight, noiseWidth))

    generatedImage = generator(noise)  
    plotter.plotImage(generatedImage)
    plotter.plotImage(discriminator(generatedImage)[0])

#%%
if(training_mode):
    discriminatorLossesAveragesArray = []
    generatorLossesAveragesArray = []
    tensorReal = Tensor( full( (batchSize, 1), fill_value=1, dtype=f32 ) ).cuda()
    tensorFake = Tensor( full( (batchSize, 1), fill_value=0, dtype=f32 ) ).cuda()

#%%
noise = noiseGenerator.generate(batchSize, channels, (noiseHeight, noiseWidth))
print(generator(noise).size())
plt.imshow(t.ToPILImage()(generator(noise)[0].cpu().detach()))

#%%
print(len(training_loader))

#%%
tempDiscriminatorRatio = discriminatorIterationsRatio
tempGeneratorRatio = generatorIterationsRatio

#%% 
for epoch in range(epochs):

    if(lookup):
        break

    if(training_mode):

        noise1Array = []
        noise2Array = []        
        discriminatorLossesArray = []
        generatorLossesArray = []
        seedsArray = []

        for batch in range(len(training_loader)):
            
            seedsArray.append(uni(0.0, 1.0))

            noise = noiseGenerator.generate(batchSize, channels, (noiseHeight, noiseWidth))
            noise1Array.append(noise)
            
            noise = noiseGenerator.generate(batchSize, channels, (noiseHeight, noiseWidth))
            noise2Array.append(noise)   

            discriminatorIterationLosses = []

        for imageIndex, image in enumerate(training_loader, 0):
            
            if(imageIndex % tempDiscriminatorRatio == 0):            
                random = uni(0.00, 1.00)

                randomTensor = noise1Array[imageIndex]
                generatedImage = generator(randomTensor)  

                if(random >= 0.05):
                    guess_real = discriminator(image)
                    guess_fake = discriminator(generatedImage)
                else:
                    guess_real = discriminator(generatedImage)
                    guess_fake = discriminator(image)

                loss_d_real = criterion_d(guess_real, tensorReal )
                loss_d_fake = criterion_d(guess_fake, tensorFake ) 

                loss_d = loss_d_real + loss_d_fake
                discriminatorLossesArray.append(loss_d.item())

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

            if(imageIndex % tempGeneratorRatio == 0):

                randomTensor = noise2Array[imageIndex]
                generatedImage = generator(randomTensor)

                loss_g = criterion_g( discriminator(generatedImage), tensorReal )
                generatorLossesArray.append(loss_g.item())
                
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

            if(loss_g):
                generatorLossesArray.append(loss_g.item())
                # if(loss_d.item() < loss_g.item()):
                #     tempGeneratorRatio = 1
                #     tempDiscriminatorRatio = discriminatorIterationsRatio
                # else:
                #     tempDiscriminatorRatio = 1
                #     tempGeneratorRatio = generatorIterationsRatio
            else:
                generatorLossesArray.append(0)

        discriminatorLossesAveragesArray.append(sum(discriminatorLossesArray) / len(discriminatorLossesArray))
        generatorLossesAveragesArray.append(sum(generatorLossesArray) / len(generatorLossesArray))

        if(epoch % epochsPerPrint == 0):

            save(generator.state_dict(), './generator.pth')
            save(discriminator.state_dict(), './discriminator.pth')

            print("Epoch: ", epoch)

            plotter.plotImage(generatedImage)
            plotter.plotImage(image)
            plotter.multiGraph([discriminatorLossesAveragesArray, generatorLossesAveragesArray], ['r', 'w'], epochsPerPrint)
    else:
        noise = noiseGenerator.generate(batchSize, channels, (noiseHeight, noiseWidth))

        generatedImage = generator(noise)  
        plotter.plotImage(generatedImage)
    

#%%
if(training_mode):
    save(generator.state_dict(), './generator.pth')
    save(discriminator.state_dict(), './discriminator.pth')
