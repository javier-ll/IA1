import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from medicion import *
from mpl_toolkits.mplot3d import Axes3D

class KMeansClustering:
    def __init__(self,k=3):
        self.k=k
        self.centroides=None
    
    @staticmethod
    def distancia_euclidiana(punto_dato, centroides):
        return np.sqrt(np.sum((centroides-punto_dato)**2, axis=1))
    def fit(self,X,indices_seleccionados,iteraciones_max=200):
        self.centroides = np.empty((self.k, 10))
        #self.centroides=np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0))
        self.centroides[0]=[2.19813262e-03, 8.31325680e-08, 1.54576883e-09, 3.84474836e-09,
                            8.64757395e-18, 1.14845687e-13, 3.61535442e-18, 1.05852417e+00,
                            8.77275960e-01, 1.00481841e+00]
        self.centroides[1]=[ 4.61136120e-03,  1.43170724e-05,  5.92379493e-09,  2.86788554e-09,
                            -4.05912354e-19, -4.31901129e-12,  1.18137159e-17,  2.47867299e+00,
                             1.37026198e-01,  6.87215573e+00]
        self.centroides[2]=[ 7.46113357e-03, 5.03785208e-05, 1.82905453e-09, 6.18684916e-09,
                             2.06360068e-17, 4.34635459e-11, 2.70224947e-18, 7.95724466e-01,
                             1.26312008e-01, 3.14797903e+01]
        self.centroides[3]=[ 8.24795508e-04, 1.43647927e-10, 1.10900196e-12, 1.47910461e-12,
                             1.88023351e-24, 1.67709379e-17, 2.30985853e-25, 1.00209644e+00,
                             9.94098100e-01, 1.00619638e+00]
        self.centroides = self.centroides[:, indices_seleccionados]
        self.centroides = normalizar(self.centroides, datos_max, datos_min)
        for _ in range(iteraciones_max):
            y=[]
            for punto_dato in X:
                distancias=KMeansClustering.distancia_euclidiana(punto_dato, self.centroides)
                # Índice de la mínima componente (cluster al que pertenece el respectivo punto_dato)
                numero_de_cluster=np.argmin(distancias)
                y.append(numero_de_cluster)
            
            y=np.array(y) # Vector con lista de cluster correspondiente de cada punto_dato

            indices_cluster=[]

            for i in range(self.k):
                # Almacena como arrays los índices de los datos correspondiente a cada cluster
                indices_cluster.append(np.argwhere(y==i))

            centros_cluster=[]

            for i, indices in enumerate(indices_cluster):
                if len(indices)==0: # No hay puntos asignados a ese cluster
                    # Almacena el vector de características correspondiente a cada cluster
                    centros_cluster.append(self.centroides[i])
                else:
                    # Almacena la media entre los puntos pertenecientes a indices
                    centros_cluster.append(np.mean(X[indices], axis=0)[0])
            # Calcula la máxima diferencia entre los dos vectores
            if np.max(self.centroides-np.array(centros_cluster))<0.0001:
                print("Se llegó a la mínima diferencia")
                break
            if _==iteraciones_max:
                print("Se llegó a la máxima iteración")
                self.centroides=np.array(centros_cluster)
            else:
                self.centroides=np.array(centros_cluster)
        return y

# Cargar imágenes de base de datos
def cargar_imagenes(directorio):
    imagenes = []
    for archivo in os.listdir(directorio):
        ruta = os.path.join(directorio, archivo)
        thresh_, contorno=preprocesar_imagen(ruta, False, 1)
        HuM=calcular_parametros_y_hu(thresh_, contorno)
        if thresh_ is not None:
            imagenes.append(HuM)  # Agrega el vector de características
    return imagenes

def preprocesar_imagen(ruta, graficar, tipo):
    img = cv2.imread(ruta)
    # La convertimos a escala de grises.
    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Filtro Gausiano.
    imagen_gauss = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    # Umbralizacion de la imagen.
    if tipo==1 or tipo==2:              # Base de datos y objeto de referencia
        _, imagen_binaria = cv2.threshold(imagen_gauss, 107, 255, cv2.THRESH_BINARY)
    if tipo==3 or tipo==4:              # Objeto a clasificar y objeto a medir
        _, imagen_binaria = cv2.threshold(imagen_gauss, 120, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_max = max(contours, key=cv2.contourArea)

    if graficar:
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))
        fig.suptitle("Preprocesamiento de imágenes", fontsize=16)
        
        axs[0].imshow(img)
        axs[0].set_title('Imagen original')
        axs[0].axis('off')
        
        axs[1].imshow(imagen_gris)
        axs[1].set_title('Imagen en grises')
        axs[1].axis('off')

        axs[2].imshow(imagen_gauss)
        axs[2].set_title('Filtro Gaussiano')
        axs[2].axis('off')
        
        axs[3].imshow(imagen_binaria, cmap='gray')
        axs[3].set_title('Imagen binaria')
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()
    
    if tipo == 1:
        return imagen_binaria, contour_max
    elif tipo == 2:
        return img, imagen_binaria
    elif tipo == 3:
        return imagen_binaria, contour_max
    elif tipo == 4:
        return img, imagen_binaria


def calcular_parametros_y_hu(imagen, contorno_maximo):
    
    # Aproximacion de contorno del objeto a figura geometrica.
    epsilon = 0.001 * cv2.arcLength(contorno_maximo, True)
    aproximacion = cv2.approxPolyDP(contorno_maximo, epsilon, True)

    momentos = cv2.moments(imagen)
    momentosHu = cv2.HuMoments(momentos)

    A = cv2.contourArea(aproximacion)
    P = cv2.arcLength(aproximacion,True)
    # Calcular relación de aspecto
    x, y, w, h = cv2.boundingRect(contorno_maximo)
    relacion_aspecto = float(w) / h
    
    # Calcular circularidad
    if P == 0:
        circularidad = 0
    else:
        circularidad = (4 * np.pi * A) / (P * P)
    
    # Calcular elongación
    if len(contorno_maximo) >= 5:  # fitEllipse requiere al menos 5 puntos
        ellipse = cv2.fitEllipse(contorno_maximo)
        (a, b) = ellipse[1]  # Obtener los ejes de la elipse
        if a>b:
            elongation = a / b 
        else:
            elongation = b / a
    else:
        print("No hay suficientes puntos para la elongación")
        elongation = None  # No se puede calcular la elongación si no hay suficientes puntos
    
    # Crear el vector de características
    if elongation is not None:
        vector_caracteristicas = np.append(momentosHu, [relacion_aspecto, circularidad, elongation])
    else:
        vector_caracteristicas = np.append(momentosHu, [relacion_aspecto, circularidad])
    
    return vector_caracteristicas

def normalizar(datos, maximo, minimo):   
    # Obtener los valores mínimos y máximos de cada característica en X
    return (datos-minimo)/(maximo-minimo)

# Directorio que contiene las imágenes de tuercas
directorio_tuercas = "Tuercas/"
tuercas = cargar_imagenes(directorio_tuercas)
# Directorio que contiene las imágenes de tornillos
directorio_tornillos = "Tornillos/"
tornillos = cargar_imagenes(directorio_tornillos)
# Directorio que contiene las imágenes de clavos
directorio_clavos = "Clavos/"
clavos = cargar_imagenes(directorio_clavos)
# Directorio que contiene las imágenes de arandelas
directorio_arandelas = "Arandelas/"
arandelas = cargar_imagenes(directorio_arandelas)

datos = []
datos = np.concatenate((tuercas, tornillos, clavos, arandelas), axis=0)
indices_seleccionados=[8, 1, 5, 9]
datos = datos[:, indices_seleccionados]
datos_max=np.max(datos, axis=0)
datos_min=np.min(datos, axis=0)
datos = normalizar(datos, datos_max, datos_min)
kmeans=KMeansClustering(k=4)

etiquetas=kmeans.fit(datos, indices_seleccionados)

# Cargar la imagen a clasificar
imagen_path="clavo.jpg"
thresh_,contorno= preprocesar_imagen(imagen_path, True, 3)
dato_analizar=calcular_parametros_y_hu(thresh_,contorno)
dato_analizar = dato_analizar[indices_seleccionados]
dato_analizar = normalizar(dato_analizar, datos_max, datos_min) 
# Calcular la distancia de la imagen a los centroides
distancias = kmeans.distancia_euclidiana(dato_analizar, kmeans.centroides)
# Obtener la etiqueta del centroide más cercano (cluster)
etiqueta_predicha = np.argmin(distancias)
# Mapear la etiqueta predicha a la categoría correspondiente
categorias = ["Tuerca", "Tornillo", "Clavo", "Arandela"]
categoria_predicha = categorias[etiqueta_predicha]

print("La imagen pertenece a la categoría:", categoria_predicha)

#Medida de referencia
referencia_path="referencia.jpg"
referencia_original, referencia_filtrada = preprocesar_imagen(referencia_path, False, 2)
referencia_original = cv2.resize(referencia_original, (300, 250))
referencia_filtrada = cv2.resize(referencia_filtrada, (300, 250))
objeto_original, objeto_filtrado = preprocesar_imagen(imagen_path, False, 4)
objeto_original = cv2.resize(objeto_original, (300, 250))
objeto_filtrado = cv2.resize(objeto_filtrado, (300, 250))
# Calcular medida de clavo o tornillo
if (categoria_predicha==categorias[1] or categoria_predicha==categorias[2]):
    height, width = medicion(referencia_original, referencia_filtrada, objeto_original, objeto_filtrado)
    if height>width:
        medida = height
    else:
        medida = width
    print("El ", categoria_predicha, "mide: ", medida, " cm.")

# Graficar
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colores = ['r', 'g', 'b', 'y']
for i in range(kmeans.k):
    ax.scatter(datos[etiquetas == i, 0], datos[etiquetas == i, 1], datos[etiquetas == i, 2], c=colores[i], label=f'{categorias[i]}')

ax.scatter(kmeans.centroides[:, 0], kmeans.centroides[:, 1], kmeans.centroides[:, 2], c=colores, marker='*', s=200, label='Centroides')
ax.scatter(dato_analizar[0], dato_analizar[1], dato_analizar[2], c='purple', marker='o', s=100, label='Dato a analizar')

ax.set_xlabel('Circularidad')
ax.set_ylabel('2do Momento de Hu')
ax.set_zlabel('6to Momento de Hu')
ax.set_title('Gráfico 3D de los datos y centroides')
ax.legend()

plt.show()
