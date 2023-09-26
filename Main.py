import resnet50
import numpy
import math
from matplotlib import pyplot as plt



def CP(*matrix):
    dtype = numpy.result_type(*matrix)
    length = len(matrix)
    array = numpy.empty([len(value) for value in matrix] + [length], dtype=dtype)
    for i, value in enumerate(numpy.ix_(*matrix)):
        array[..., i] = value
    return array.reshape(-1, length)



def computeDistance(features, window):
    distances = {}
    for i in range(window):
        distance = []
        indexes = []
        for j in range(window):
            distance.append(numpy.linalg.norm(features[i] - features[j]))
            indexes.append(j)

        distance, indexes = zip(*sorted(zip(distance, indexes)))
        distances[i] = distance, indexes
    return distances


def rankNormalizationAndHypergraphCreation(distances, window):
    sortedIndexes = {}

    for i in range(window):
        vertices = []
        indexes = []
        hyperedges = []
        dI = distances[i]
        for j in range(window):
            dJ = distances[dI[1][j]]
            normalized_distance = ((j + dJ[1].index(i)) / 2)
            hyperedges.append(normalized_distance)
            indexes.append(dI[1][j] + 1)
            vertices.append("V" + str(1 + dI[1][j]))

        hyperedges, vertices, indexes = zip(*sorted(zip(hyperedges, vertices, indexes)))
        sortedIndexes[i] = indexes

    return hyperedges, sortedIndexes, indexes



def rankNormalizationAndHypergraphCreation2(distances, window):
    sortedIndexes = {}

    for i in range(window):
        distanceI = distances[i]
        indexes = []
        vertices = []
        hyperedges = []
        for j in range(window):
            distanceJ = distances[distanceI[1][j] - 1]
            normalized_distance = ((j + list(distanceJ[1]).index(i + 1)) / 2)
            hyperedges.append(normalized_distance)
            vertices.append("V" + str(1 + distanceI[1][j]))
            indexes.append(distanceI[1][j])


        hyperedges, vertices, indexes = zip(*sorted(zip(list(hyperedges), list(vertices), list(indexes))))
        sortedIndexes[i] = indexes

    return hyperedges, sortedIndexes, indexes




def weightCompute(sortedIndexes, window):
    weights = {}
    sumOfWeights = {}
    for i in range(window):
        tempW = []
        sum = 0
        for j in range(window):
            value = (math.log(sortedIndexes[i][j], 10))
            value = 1 - value
            sum = sum + value
            tempW.append(value)
        weights[i] = tempW
        sumOfWeights[i] = sum
    return weights, sumOfWeights


def computeSandC(hyperedges, sortedIndexes, indexes, window, weights, sumOfWeights):
    hyperedges = numpy.array(hyperedges)
    S = CP(hyperedges, hyperedges)
    S = numpy.multiply(S, S)


    indexOfS = {}
    for i in range(window):
        indexOfS[i] = S[i:i + window], sortedIndexes[i]


    C = {}
    for i in range(window):
        temp = []
        for j in range(window):
            temp.append(sumOfWeights[i] * weights[i][j] * weights[i][i])
        C[i] = temp


    W = {}

    for i in range(window):
        tempW = []
        tempC = numpy.array(C[i])
        for j in range(window):
            tempS = numpy.array(indexOfS[i][0][j][1])
            multiplication = numpy.multiply(tempC, tempS)
            tempW.append(numpy.sum(multiplication))
        tempW, indexes = zip(*sorted(zip(tempW, indexes)))
        W[i] = tempW, indexes

    return W




def norm(W, previousW):
    return numpy.linalg.norm((numpy.array(W[0][1][:15]) - numpy.array(previousW[0][1][:15])))




def getImagesAndTheirClass(X_train, window):
    imageClass = {}
    image, labels = X_train.next()
    for i in range(window):
        imageClass[i + 1] = labels[i]
    return image, imageClass




def convert(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = numpy.clip(x, 0, 255).astype('uint8')
    return x




def recreate(image, weights, window):
    images = []
    for i in range(window):
        if i + 1 in list(weights[0][1]):
            tempImg = numpy.array(image[i])
            images.append(convert(tempImg))
    return images




def plotResults(images):
    plt.figure(figsize=(20, 20))
    plt.suptitle("The closest images for the first image in order", fontsize=15)
    for i in range(10):
        x = plt.subplot(8, 8, i + 1)
        plt.imshow(list(images)[i])
        plt.axis("off")
    plt.show()



# ΜΑΙΝ

window = 100


features, X_train = resnet50.getImageFeaturesAndTrainSet()


image, imageClass = getImagesAndTheirClass(X_train=X_train, window=window)


distances = computeDistance(features=features, window=window)


hyperedges, sortedIndexes, indexes = rankNormalizationAndHypergraphCreation(distances=distances, window=window)


weights, sumOfWeights = weightCompute(sortedIndexes=sortedIndexes, window=window)


W = computeSandC(hyperedges=hyperedges, sortedIndexes=sortedIndexes, indexes=indexes, window=window, weights=weights,
                 sumOfWeights=sumOfWeights)

iterations = 0
difference = float('inf')
threshold = 75
differenceThreshold = 170
previousDifference = None
previousW = W


while difference > threshold:

    previousDifference = difference
    if iterations != 0:
        previousW = W

 
    hyperedges, sortedIndexes, indexes = rankNormalizationAndHypergraphCreation2(distances=previousW, window=window)


    weights, sumOfWeights = weightCompute(sortedIndexes=sortedIndexes, window=window)

    W = computeSandC(hyperedges=hyperedges, sortedIndexes=sortedIndexes, indexes=indexes, window=window, weights=weights,
                     sumOfWeights=sumOfWeights)


   
    difference = norm(W=W, previousW=previousW)
    print(" The value between the difference and the threshold is " + str(difference))


    if difference < differenceThreshold:
        break

    iterations = iterations + 1


images = recreate(image=image, weights=W, window=window)
plotResults(images=images)
