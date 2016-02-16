import numpy as np
from scipy.special import expit
import scipy.io.matlab as mio
import time

class RBM:
        def __init__(self, numHidden, numVisible,
                    learningRate, numEpochs = 1000, numGibbsSteps = 1,
                    verbose = True, batchSize = 20, methodVisible = "binary",
                    methodHidden = "binary", numCopies = 10,
                    weightVariance = 0.1):
                """__init__: Initializes the RBM from the given input."""
                self.numHidden = numHidden
                self.numVisible = numVisible
                self.learningRate = learningRate
                self.numEpochs = numEpochs
                self.numGibbsSteps = numGibbsSteps
                self.verbose = verbose #for displaying output
                self.batchSize = batchSize
                self.methodHidden = methodHidden
                self.methodVisible = methodVisible
                self.numCopies = numCopies
                if (numCopies != 1 and (methodVisible != "binomial" or methodHidden != "binomial")):
                        print("Warining: numCopies > 1")
                self.weightVariance = weightVariance
                # initialize weight matrix
                self.W = self.weightVariance * np.random.randn(self.numHidden + 1,self.numVisible + 1)
                self.errors = []
        def PreprocessData(self, xTrain):
                """PreprocessData(self,xTrain): converts the data to range of visible units."""
                numRows = xTrain.shape[0]
                numCols = xTrain.shape[1]
                # convert data to appropriate range for input vectors
                if self.methodVisible == "binary":
                        #convert units to binary, read that this was a good rule somewhere
                        xTrain[xTrain < 35] = 0
                        xTrain[xTrain > 0] = 1
                elif self.methodVisible == "binomial":
                        #convert visible units to have integer values in range [0,numCopies]
                        xTrain = np.round(xTrain / 255 * self.numCopies)
                elif self.methodVisible == "rectified":
                        #rectified units can take range on [0,\infty] so leave them alone
                        #xTrain = xTrain / 255
                        pass
                self.rescaledData = xTrain
                #padd with 1's for bias units
                xTrain = np.insert(xTrain,xTrain.shape[1],1,axis = 1)
                return xTrain
        def Rebatch(self,data):
                """Rebatch(self,data): Shuffles and rebatches data into mini-batches for training."""
                numRows = data.shape[0]
                numCols = data.shape[1]
                np.random.shuffle(data)
                numBatches = int(np.ceil(numRows/self.batchSize))
                dataOut = np.array_split(data,numBatches)
                return dataOut
        
        def Train(self, xTrain):
                """Train(self,Xtrain): Trains the rbm on the training data xTrain."""
                xTrain = self.PreprocessData(xTrain)
                timeStart = time.time()
                for epoch in range(self.numEpochs):
                        #reshuffle data each epoch
                        dataBatches = self.Rebatch(xTrain)
                        epochError = 0;
                        for batch in dataBatches:
                                #looping over the batches
                                vPlus, vMinus = self.GibbsSampling(batch)
                                vPlusProbs = self.ProbHGivenV(vPlus)
                                vMinusProbs = self.ProbHGivenV(vMinus)
                                #update W
                                self.W += self.learningRate * ( vPlusProbs @ vPlus -
                                                            vMinusProbs @ vMinus) / (batch.shape[0] * self.numCopies)

                                # R_MSE
                                epochError += np.sum((vPlus - vMinus) ** 2)

                        epochError = np.sqrt(epochError/(xTrain.shape[0]*self.numCopies))
                        self.errors.append(epochError)
                        if self.verbose:
                                print('Epoch %s of %s' % (epoch,self.numEpochs))
                                print('      error %s' % (epochError))
                timeFinish = time.time()
                self.timeRunning = timeFinish - timeStart
                print("Completed training in %s seconds" % self.timeRunning)
        def GibbsSampling(self, batch, steps = None):
                """ GibbsSampling(batch): performs steps of Gibbs sampling
                        to produce the point estimate.
                        """
                if steps is None:
                        steps = self.numGibbsSteps
                vPlus = batch #the training sample, the start of the Gibbs sampling chain
                vMinus = vPlus
                for k in range(steps):
                        hPlus, hPlusProbs = self.SampleHGivenV(vMinus)
                        vMinus, vMinusProbs = self.SampleVGivenH(hPlus)
                return vPlus, vMinus

        # h given v
        def SampleHGivenV(self, v):
                """SampleHGivenV(self, v): Samples the hidden units given the visible states."""
                probH = self.ProbHGivenV(v)
                if self.methodHidden == "binary":
                        h = (probH > np.random.rand(probH.shape[0],probH.shape[1]))*1
                elif self.methodHidden == "binomial":
                        h = np.round(self.numCopies * probH)
                elif self.methodHidden == "rectified":
                        h = softMax(probH + np.random.randn(probH.shape[0], probH.shape[1]))
                else:
                        raise NameError(method)
                h[-1,:] = 1
                return h, probH

        def ProbHGivenV(self, v):
                """ProbHGivenV(self, v): Returns the probabilities of activation for the hidden units 
                given the visible states."""
                if self.methodHidden == "binary":
                        probH = expit(self.W@v.T)
                elif self.methodHidden == "binomial":
                        probH = expit(self.W@v.T)
                elif self.methodHidden == "rectified":
                        probH = softMax(self.W@v.T)
                else:
                        raise NameError(methodHidden)
                #probH[-1,:] = 1
                return probH

        #v given h
        def SampleVGivenH(self, h):
                """SampleHGivenV(self, v): Samples the visible units given the hidden states."""
                probV = self.ProbVGivenH(h)
                if self.methodVisible == "binary":
                        v = (probV > np.random.rand(probV.shape[0],probV.shape[1]))*1
                elif self.methodVisible == "binomial":
                        v = np.round(self.numCopies * probV)
                elif self.methodVisible == "rectified":
                        v = softMax(probV+np.random.randn(probV.shape[0],probV.shape[1]))
                else:
                        raise NameError(method)
                v[:,-1] = 1
                return v, probV
        def ProbVGivenH(self, h):
                """ProbVGivenH(self, v): Returns the probabilities of activation for the  units visible
                given the hidden states."""
                if self.methodVisible== "binary":
                        probV = expit(h.T@self.W)

                elif self.methodVisible == "binomial":
                        probV = expit(h.T@self.W)
                elif self.methodVisible == "rectified":
                        probV = softMax(h.T@self.W)
                else:
                        raise NameError(method)
                #probV[-1,:] = 1
                return probV

        def Daydream(self,data,steps = None):
                """Daydream(self,data, steps = None): Performs steps steps of Gibs sampling on
                the trained RBM and returns the visible states after the chain is complete."""
                if steps is None:
                        steps = self.numGibbsSteps
                vPlus, vMinus = self.GibbsSampling(data,steps)
                return vMinus

def softMax(x):
        """softMax(x) returns component-wise max(x,0)."""
        x[x<0] = 0
        return x


def LoadMNIST(subsampling = 0):
        """LoadMNIST(subsampling = 0): Loads and organizes MNIST data for this script.
            This function also returns the data labels, so that the output can be 
            used for testing feature vectors in classification algorithms."""
        #load mnist data
        mFile = mio.loadmat('mnist_all.mat')

        #trainingData
        trainingKeys = [key for key in mFile.keys() if 'train' in key]
        trainingDataSubsets = []
        i = 0
        trainingLabels = []
        for key in trainingKeys:
                trainingDataTmp = mFile[key]
                trainingDataSubsets.append(trainingDataTmp)
                trainingLabels.append(np.ones(trainingDataTmp.shape[0])*i)
                i += 1
        trainingLabels = np.concatenate(trainingLabels,axis=0)
        rawTrainingData = np.concatenate(trainingDataSubsets, axis=0)
        rawTrainingData = np.insert(rawTrainingData, rawTrainingData.shape[1], trainingLabels, axis = 1)

        #testingData
        testingKeys = [key for key in mFile.keys() if 'test' in key]
        testingDataSubsets = []
        i = 0
        testingLabels = []
        for key in testingKeys:
                testingDataTmp = mFile[key]
                testingDataSubsets.append(testingDataTmp)
                testingLabels.append(np.ones(testingDataTmp.shape[0])*i)
                i += 1
        testingLabels = np.concatenate(testingLabels,axis=0)
        rawTestingData = np.concatenate(testingDataSubsets, axis=0)
        rawTestingData = np.insert(rawTestingData, rawTestingData.shape[1], testingLabels, axis = 1)

        #subsampling
        if subsampling > 0:
                rawTestingData = rawTestingData[[i for i in range(0,rawTestingData.shape[0],subsampling)],:]
                rawTrainingData = rawTrainingData[[i for i in range(0,rawTrainingData.shape[0],subsampling)],:]
        #shuffling needs to be done post subsampling for slower machine
        np.random.shuffle(rawTestingData)
        np.random.shuffle(rawTrainingData)
        imageShape = [28,28]
        return rawTrainingData, rawTestingData, imageShape

def TileImages(data,imshape = None):
    """TileImages(data,imshape): Produces a nice littel plot of the rows of a data
        matrix as images."""

if __name__ == '__main__':
        #load data
        print("loading data")
        rawTrainingData, rawTestingData, imageShape = LoadMNIST(100)
        testData = rawTestingData[:,:-1]
        trainingData = rawTrainingData[:,:-1]
        # initialize rbm
        print("initializing rbm")
        rbm = RBM(numHidden = 100,
                numVisible = imageShape[0]*imageShape[1],
                learningRate = 0.01,
                weightVariance = 0.01,
                numEpochs = 8,
                batchSize = 20,
                numGibbsSteps = 1,
                numCopies = 1,
                methodVisible = "rectified",
                methodHidden = "binary")
        # train rbm
        print("training rbm")
        rbm.Train(trainingData)


