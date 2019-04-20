
![](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Alexandra-Raibolt-Avatar.png)

<div style="text-align:center">
**ALEXANDRA RAIBOLT**
</div>

<div style="text-align:center">
MSc Student @ IME​ and​ BI & Data Management @ Firjan
</div>

# Construa sua própria Rede Neural Convolucional em 3, 2, 1...!

<div style="text-align:justify">Uma introdução as Redes Neurais Convolucionais em Python, usando o framework TensorFlow com Keras.</div>

## Visão Geral

<div style="text-align:justify">Este Jupyter Notebook mostra passo a passo, o processo de construção de uma Rede Neural Convolucional para reconhecimento e clasificação de Dígitos Manuscritos em Python, usando o framework TensorFlow com Keras.</div>
<br>
<div style="text-align:justify">Neste exemplo, usamos o conjunto de dados [MNIST](http://yann.lecun.com/exdb/mnist "MNIST").</div>
<br>
<div style="text-align:justify">**Aviso prévio:**</div>
<br>
<div style="text-align:justify">O modelo de Redes Neurais Convolucional proposto nesta palestra foi implementado em Python (versão 5.4.0) usando o framework TensorFlow (versão 1.4.0) com Keras (versão 2.2.4) usando uma arquitetura baseada em GPU, e pode não funcionar com outras versões.</div>

## Requisitos

- Python 3 (versão 5.4.0);
- TensorFlow (versão 1.4.0);
- Keras (versão 2.2.4);
- Jupyter Notebook (versão 4.4.0).

## Dependências

- matplotlib;
- numpy.

<div style="text-align:justify">Você pode instalar dependências ausentes com [pip](https://pip.pypa.io/en/stable "pip"). E instalar o TensorFlow via [TensorFlow](https://www.tensorflow.org/install "TensorFlow") link.</div>

## O que é Aprendizado de Máquina?

<div style="text-align:justify">Aprendizado de Máquina (em inglês, *Machine Learning*) é uma subárea da Inteligencia Artifical que tem como propósito automatizar máquinas através da criação de modelos analíticos. Tal ideia se baseia na premissa de que máquinas tem a habilidade de aprender, tomar decisões, extrair *insights* e reconhecer padrões (através de dados) sem a necessidade de serem explicitamente programadas, ou seja, sem interferência humana.</div>

## Tipos de Aprendizado de Máquina:

![Tipos de Aprendizado de Maquina](https://raw.githubusercontent.com/whoisraibolt/MNIST-Convolutional-Neural-Network/master/images/Tipos-de-Aprendizado-de-Maquina.png)

# Aprendizado Supervisionado:

<div style="text-align:justify">Um conjunto de dados de entradas e suas respectivas saídas desejadas (denominadas como classes) são fornecidas por um “professor” ao modelo/computador, mostrando que há relação entre os dados de entrada e de saída.</div>
<br>
<div style="text-align:justify">O objetivo deste tipo de aprendizado está em mapear as entradas para as suas respectivas saídas desejadas ao encontrar uma regra geral que seja apta a classificar novos dados entre as classes já presentes no conjunto de dados.</div>

## Classificação:

<div style="text-align:justify">Em uma tarefa de classificação, o objetivo é mapear os tipos de entradas e categoriza-las em classes distintas, ou seja, realiza previsões onde os resultados de saída são de natureza discreta.</div>
<br>
<div style="text-align:justify">Algoritmos de classificação podem ser usados para problemas como: classificação de gênero, classificação de imagens médicas, classificação de expressões faciais, etc.</div>
<br>
<div style="text-align:justify">K-vizinhos mais próximos, Máquinas de Vetores de Suporte e Redes Neurais Convolucionais são exemplos de algoritmos de classificação.</div>

## O que é Rede Neural Artificial?

<div style="text-align:justify">Tendo como exemplo, a identificação de objetos e ambientes distintos, é questionado como o cérebro humano é capaz de traduzir uma imagem na retina humana para algo onde um indivíduo consiga identificar e classificar.</div>
<br>
<div style="text-align:justify">Compreendendo toda a complexidade do cérebro humano, tornou-se necessário o estudo e compreensão de como os sistemas neurais biológicos funcionam, e a partir de tais estudos, pesquisadores foram capazes de proporcionar a estrutura básica para a construção de modelos de Redes Neurais Artificiais (em inglês, *Artificial Neural Networks - ANN*).</div>
<br>
<div style="text-align:justify"></div>

## O que é Rede Neural Convolucional?

<div style="text-align:justify">As diversas arquiteturas de Redes Neurais Convolucionais são aplicadas com sucesso em desafios de reconhecimentoe classificação de imagens, embora sejam também utilizadas em processamento de vídeo, e Processamento de Linguagem Natural (em inglês, *Natural Language Processing*).</div>
<br>
<div style="text-align:justify">Uma Rede Neural Convolucional pode ser caracterizada como sendo uma ANN de múltiplas camadas onde primariamente faz-se a suposição de que os dados de entrada sejam imagens.</div>
<br>
<div style="text-align:justify">As Redes Neurais Convolucionais hoje, são consideradas estado-da-arte para a resolução de problemas na área de Visão Computacional, possuindo uma variedade de modelos e de arquiteturas (AlexNet, Inception, ResNet, etc.)</div>
<br>
<div style="text-align:justify">Uma arquitetura básica de CNN pode ser dividida em camadas.</div>

## Arquitetura de uma Rede Neural Convolucional

![Arquitetura de uma Rede Neural Convolucional](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Arquitetura-CNN.png)

## Camada de Entrada (ou em inglês, Input Layer)

<div style="text-align:justify">Onde a imagem (ou base de imagens) é inserida na rede.</div>
<br>
<div style="text-align:justify">Computacionalmente falando, toda imagem pode ser representada como sendo uma matriz de valores de pixel, como pode ser visto abaixo:</div>

![Camada de Entrada](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Camada-de-Entrada.png)

<div style="text-align:justify">Portanto, ao receber uma imagem em escala de cinza como entrada, o computador "enxergará" uma matriz de valores de pixel (*m* x *n* x *1*), onde o valor 1 refere-se a um componente pré-estabelecido de uma imagem – Canal - neste caso o valor 1 refere-se a uma imagem em escala de cinza. Portanto, será atribuído valores de intensidade de pixels dentro do intervalode 0 à 255 (onde 0 representa pontos pretos e 255 representa pontos brancos) para cada ponto da matriz.</div>

## Mãos a obra!


```python
# Imports
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import numpy as np
```

    Using TensorFlow backend.



```python
# Dimensions of the data

# We know that images are 28 pixels in each dimension
in_height = 28
in_width = 28

# Tuple with height and width of images used to reshape arrays
image_shape = (in_height, in_width)

# We know MNIST has 60.000 training set images
num_data_train_images = 60000

# We know MNIST has 10.000 training set images
num_data_test_images = 10000

# Classes info
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Number of classes
num_classes = len(classes)

# Number of colour channels for the images: 1 channel for gray-scale
# Channels mean number of primary colors
num_channels = 1
```


```python
# Load Dataset

# The data, split between train and test sets
# The datasets are 3D arrays: (batch, height, width)
# Returns the values of the function for images_set, labels_set on train-set
# Returns the values of the function for images_set, labels_set on test-set
(data_train_images, data_train_labels), (data_test_images, data_test_labels) = mnist.load_data()

# Our CNN accepts only 4D arrays: (batch, height, width, channels)

# Reshape data_train_images 
data_train_images = data_train_images.reshape(num_data_train_images, in_height, in_width, num_channels)

# Reshape data_test_images
data_test_images = data_test_images.reshape(num_data_test_images, in_height, in_width, num_channels)
```


```python
print("Size of:")
print("- Training-set:\t\t{}".format(len(data_train_images)))
print("- Test-set:\t\t{}".format(len(data_test_images)))
```

    Size of:
    - Training-set:		60000
    - Test-set:		10000


## Codificação One-Hot

<div style="text-align:justify">As classes que representam os dez digitos manuscritos são representados por valores numéricos.</div>
<br>
<div style="text-align:justify">Utilizar valores numéricos para representar rótulos categóricos em algoritmos de ML não é considerado como sendo uma boa prática, pois é possível que estes valores gerados possam prejudicar e influenciar a efetividade do algoritmo através do processo de aprendizado.</div>
<br>
<div style="text-align:justify">Na presença desta restrição, convertemos os valores numéricos que representam os rótulos categóricos (tanto os valores numéricos do conjunto de treinamento quanto os valores numéricos do conjunto de teste) para vetores binários que representam cada um dos valores numéricos, através da codificação One-Hot.</div>
<br>
<div style="text-align:justify">A codificação One-Hot é uma codificação de estados que consiste em gerar $ n $ vetores binários para $ n $ valores numéricos que representam cada um dos rótulos categóricos. Deste modo, One-Hot realiza a codificação para cada estado, de forma que cada valor numérico seja representado por um único vetor binário de tamanho $ n $ igual a quantidade de classes presentes no conjunto de dados.</div>
<br>
<div style="text-align:justify">Por exemplo: a saída $ 8 $, de acordo com a codificação One-Hot seria: $ [0, 0, 0, 0, 0, 0, 0, 0, 1, 0] $</div>


```python
# Class labels are One-Hot coded, meaning that each label is a vector with 10 elements,
# all of which are zero except one element
data_train_labels = keras.utils.to_categorical(data_train_labels, num_classes)
data_test_labels = keras.utils.to_categorical(data_test_labels, num_classes)

data_test_cls = np.argmax(data_test_labels, axis=1)

print(data_test_cls)
```

    [7 2 1 ..., 4 5 6]



```python
# Helper-function for catch the class
# number and print its corresponding name
def get_number(num_class):
    if num_class == 0:
        return 'Zero'
    elif num_class == 1:
        return 'One'
    elif num_class == 2:
        return 'Two'
    elif num_class == 3:
        return 'Three'
    elif num_class == 4:
        return 'Four'
    elif num_class == 5:
        return 'Five'
    elif num_class == 6:
        return 'Six'
    elif num_class == 7:
        return 'Seven'
    elif num_class == 8:
        return 'Eight'
    elif num_class == 9:
        return 'Nine'
```


```python
# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(image_shape), cmap='gray')

        # Show true and predicted classes
        # T = True, and P = Predicted
        if cls_pred is None:
            xlabel = "T: {0}".format(get_number(cls_true[i]))
        else:
            xlabel = "T: {0}, P: {1}".format(get_number(cls_true[i]), get_number(cls_pred[i]))

        # Show the classes as the label on the x-axis
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
```


```python
# Plot a few images to see if data is correct

# Get the first images from the test-set
images = data_test_images[0:9]

# Get the true classes for those images
cls_true = data_test_cls[0:9]

# Plot the images and labels using our helper-function above
plot_images(images=images, cls_true=cls_true)
```


![png](output_39_0.png)



```python
# We build a sequential model
# The Sequential model is a linear stack of layers
model = Sequential()
```

## Camada de Convolução (ou em inglês, Convolutional Layer)

<div style="text-align:justify">É a principal operação que ocorre nas imagens inseridas na rede, e seu principal objetivo está em extrair recursos de imagens de entrada.<br>
    
<div style="text-align:justify">Uma imagem $ g $ é gerada pela convolução de um filtro $ f $ (comumente conhecido na literatura em inglês como kernel) com a imagem de entrada idefinida pela equação abaixo:<br>

$$ g(m,n)=\sum_{i=1}^{q}\sum_{j=1}^{q}f(i,j)I(m-i,n-j) $$

<div style="text-align:justify">onde $ q $ é a dimensão da máscara de convolução.</div>

![Convolução GIF](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Convolucao-GIF.gif)

<div style="text-align:justify">Um filtro pode ressaltar determinada característica presente em uma imagem, como por exemplo, sombras, bordas, entre outros.</div>
<br>
<div style="text-align:justify">A Figura abaixo apresenta o resultado de uma convolução aplicada a uma imagem de entrada.</div>

![Resultado Convolucao](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Resultado-Convolucao.png)

<div style="text-align:justify">Onde o resultado da convolução é comumente conhecido como um mapa de recursos (em inglês,feature maps), e onde o filtro convolucional $ f $ utilizado foi uma matriz de tamanho $ 3x3 $ como na representação a seguir:</div>

\begin{equation}
\label{equation:kernel}
f = \begin{bmatrix}
1 &0   &0 \\ 
0 &-1  &0 \\ 
0 &0   &0 
\end{bmatrix}  
\end{equation}

## Camada  ReLU (ou em inglês, ReLU Layer)

<div style="text-align:justify">É utilizada para que a rede possa convergir mais rápido, aplica para as camadas escondidas da rede uma função de ativação não linear à saída $ x $ da camada anterior, como abaixo:</div>

\begin{equation}
\label{equation:relu}
f(x) = max(0, x)
\end{equation}

<div style="text-align:justify">Portanto, a saída será $ 0 $ quando a entrada for menor que $ 0 (x < 0) $, ou a saída será $ 1 $ caso contrário. Desta forma a ativação é limitada exclusivamente a $ 0 $, logo, a função de ativação ReLU contribui em uma etapa de treinamento da rede mais rápida para arquiteturas profundas</div>


```python
# Convolutional Layer 1

# Convolution filters are 3 x 3 pixels
filter_height_1 = 3
filter_width_1 = 3

# There are 8 of these filters
num_filters_1 = 32

# Convolutional Layer 2

# Convolution filters are 3 x 3 pixels
filter_height_2 = 3
filter_width_2 = 3

# There are 16 of these filters
num_filters_2 = 64
```


```python
# Convolutional Layer 1
model.add(Conv2D(filters=num_filters_1,
                 kernel_size=(filter_height_1, filter_width_1),
                 activation='relu',
                 input_shape=(in_height, in_width, num_channels)))
```


```python
# Convolutional Layer 2
model.add(Conv2D(filters=num_filters_2,
                 kernel_size=(filter_height_2, filter_width_2),
                 activation='relu'))
```

## Camada de Subamostragem (ou em inglês, Pooling Layer)

<div style="text-align:justify">Reduz a dimensionalidade dos feature maps que serão utilizados pelas camadas seguintes, entretanto, sem perder as informações mais relevantes.  Desta forma,  podemos definir a camada de subamostragem como uma convolução que possui o objetivo de reduzir o tamanho espacial da representação, de forma a contribuir com a diminuição de quantidade de parâmetros e processamento computacional da rede.</div>
<br>
<div style="text-align:justify">A função do Max-Pooling está em pegar o valor máximo dos elementos do feature maps presentes na janela $ 2x2 $ e agrupa-los como no exemplo abaixo:</div>

![Max-Pooling](https://raw.githubusercontent.com/whoisraibolt/Convolutional-Neural-Network-MNIST/master/images/Max-Pooling.png)


```python
# This is 2x2 max-pooling, which means that we
# consider 2x2 windows and select the largest value
# in each window. Then we move 2 pixels to the next window
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2)))
```


```python
# Dropout randomly switches off some neurons in
# the network which forces the data to find new paths
model.add(Dropout(0.25))
```

## Camada de Flatten (ou em inglês, Flatten Layer)

<div style="text-align:justify">Após passar pelas camadas de convolução, ReLU e subamostragem a próxima etapa de uma CNN é a classificação, ou seja, é necessário passar pela camada totalmente conectada.  A camada totalmente conectada aceita apenas como entrada, dados de dimensão 1D.</div>
<br>
<div style="text-align:justify">A camada Flatten possui o objetivo de "achatar" todo o volume de dados 3D em um único vetor 1D de características para posteriormente servir de entrada para a camada totalmente conectada.</div>


```python
# It allows the output to be processed by standard fully connected layers
model.add(Flatten())
```

## Camada totalmente conectada (ou em inglês, Fully Connected Layer)

<div style="text-align:justify">Essa camada age como um classificador, se localiza ao final da rede, geralmente é do tipo soft-max usado para classificar a imagem de entrada. Realiza tal tarefa ao reconhecer padrões que foram gerados pelas camadas anteriores. Portanto, estima a probabilidade de quais das $ n $ classes a imagem de entrada pertence.</div>


```python
# Fully-connected layer
# Number of neurons in fully-connected layer
fc_size = 128

model.add(Dense(fc_size, activation='relu'))
```


```python
# Dropout randomly switches off some neurons in
# the network which forces the data to find new paths
model.add(Dropout(0.5))
```


```python
# Normalization of class-number output with 10 neurons (number of output classes)
# and it uses softmax activation function.
# Each neuron will give the probability of that class.
#It’s a multi-class classification that’s why softmax activation function
model.add(Dense(num_classes, activation='softmax'))
```


```python
# Define the type of loss function, optimizer and the
# metrics evaluated by the model during training and test

# Cross Entropy as a Loss function
# The lower the loss, the better a model
# The loss is calculated on training and its interperation is how well the model is doing for this one set
# Loss is not a percentage. It is a summation of the errors made for each example in training  set

# AdamOptimizer as optimizer
# It is an optimization algorithm used to perform the update of weights
# in an iterative way according to the training data set
# Needs few memory requirements, in addition to being computationally efficient

# and Accuracy as metrics to improve the performance of our neural network

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
```


```python
# Training

# Epoch:
# One Epoch is when an ENTIRE dataset is passed forward
# and backward through the neural network only ONCE

# Batch Size:
#Total number of training examples present in a single batch

# Iterations:
# Iterations is the number of batches needed to complete one epoch

# Verbose:
# Integer 0, 1, or 2
# Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch

batch_size = 128
epochs = 12

history = model.fit(data_train_images, data_train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
```

    Epoch 1/12
    60000/60000 [==============================] - 27s 457us/step - loss: 1.7346 - acc: 0.8363
    Epoch 2/12
    60000/60000 [==============================] - 10s 163us/step - loss: 0.1262 - acc: 0.9625
    Epoch 3/12
    60000/60000 [==============================] - 10s 164us/step - loss: 0.0964 - acc: 0.9708
    Epoch 4/12
    60000/60000 [==============================] - 10s 162us/step - loss: 0.0776 - acc: 0.9761
    Epoch 5/12
    60000/60000 [==============================] - 10s 163us/step - loss: 0.0677 - acc: 0.9795
    Epoch 6/12
    60000/60000 [==============================] - 10s 165us/step - loss: 0.0610 - acc: 0.9808
    Epoch 7/12
    60000/60000 [==============================] - 10s 164us/step - loss: 0.0561 - acc: 0.9823
    Epoch 8/12
    60000/60000 [==============================] - 10s 164us/step - loss: 0.0486 - acc: 0.9849
    Epoch 9/12
    60000/60000 [==============================] - 10s 165us/step - loss: 0.0483 - acc: 0.9847
    Epoch 10/12
    60000/60000 [==============================] - 10s 164us/step - loss: 0.0437 - acc: 0.9860
    Epoch 11/12
    60000/60000 [==============================] - 10s 164us/step - loss: 0.0462 - acc: 0.9857
    Epoch 12/12
    60000/60000 [==============================] - 10s 163us/step - loss: 0.0404 - acc: 0.9871


## Google Cloud Platform

<div style="text-align:justify">Oferecida pelo Google, o GCP é um conjunto de ferramentas, soluções e serviços de computação em nuvem que atua na mesma infraestrutura central do Google, ao lado de seus produtos, como o YouTube e o seu próprio mecanismo de busca.</div>
<br>
<div style="text-align:justify">O GCP disponibiliza uma série de soluções para processamento computacional, rede, armazenamento de dados, banco de dados escaláveis, exploração de dados, Internet das Coisas (em inglês,Internet of Things- IoT), ML, ferramentas de gerenciamento, monitoramento, registro, diagnósticos, entre outros.</div>
<br>
<div style="text-align:justify">Este Notebook foi processado por  meio de uma instância de máquina virtual de alto desempenho na infraestrutura do Google - Google Compute Engine (GCE) - um componente Infraestrutura como Serviço (em inglês, Infrastructure as a Service- IaaS).</div>
<br> 
<div style="text-align:justify">O GCE é um recurso disponível na plataforma Google Cloud Platform (GCP). A instância de máquina virtual on demand é composta por uma GPU NVIDIA® Tesla® K80 com memória da GPU de 12GB GDDR5 e 8 vCPUs disponíveis, além de 16 GB de memória RAM e 50GB de HD e é acessada por meio da interface de linha de comandos.</div>


```python
# Test
score = model.evaluate(data_test_images, data_test_labels, verbose=0)

# Print test accuracy
print('Test accuracy:', score[1])
```

    Test accuracy: 0.9894


![Aplausos](https://github.com/whoisraibolt/Convolutional-Neural-Network-MNIST/blob/master/images/Aplausos-GIF.gif?raw=true)

## Dicas:

- [TensorFlow](https://www.tensorflow.org/ "TensorFlow");
- [Keras](https://keras.rstudio.com/index.html "Keras");
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets "An Intuitive Explanation of Convolutional Neural Networks");
- [Hvass-Labs](https://github.com/Hvass-Labs/TensorFlow-Tutorials "Hvass-Labs");
- [How to set up Tensorflow with CUDA and cuDNN on a free Google Cloud instance](https://www.youtube.com/watch?v=abEf3wQJBmE "How to set up Tensorflow with CUDA and cuDNN on a free Google Cloud instance").

## Leituras Recomendadas:

- BEALE, Russell; JACKSON, Tom. [Neural Computing-an introduction](https://bayanbox.ir/view/7901640340179926235/Neural-Computing-An-Introduction.pdf "Neural Computing-an introduction"). CRC Press, 1990.

## Referências:

- [Towards Data Science](https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f "Towards Data Science");
- [Hvass-Labs](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb "Hvass-Labs").

# Distribuição MIT

Código lançado sob a licença [MIT](https://github.com/whoisraibolt/Convolutional-Neural-Network-MNIST/blob/master/LICENSE "MIT").
