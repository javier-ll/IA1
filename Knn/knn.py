import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from graficas import *
from medicion import *

# Definir una función para calcular la distancia Euclidiana entre dos vectores
def distancia_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Definir la clase del clasificador Knn
class Knn:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predecir(self, X):
        n_samples = X.shape[0]
        predicciones = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Calcular la distancia Euclidiana entre la muestra actual y todas las muestras de entrenamiento
            distancias = np.array([distancia_euclidiana(X[i], x_train) for x_train in self.X_train])
            
            # Ordenar las distancias y obtener los índices de las k muestras más cercanas
            indices = np.argsort(distancias)[:self.k]
            
            # Obtener las etiquetas correspondientes de las k muestras más cercanas
            k_etiquetas = [self.y_train[indice] for indice in indices]
            
            # Elegir la etiqueta que aparece con más frecuencia entre las k muestras más cercanas
            predicciones[i] = max(set(k_etiquetas), key=k_etiquetas.count)
        
        return predicciones
    
    def score(self, X, y):
        predicciones = self.predecir(X)
        exactitud = np.mean(predicciones == y)
        return exactitud

# Crear matrices para las imágenes de entrenamiento y prueba procesadas
def cargar_imagenes_y_etiquetas(directorio, etiqueta):
    imagenes = []
    etiquetas = []
    for archivo in os.listdir(directorio):
        ruta = os.path.join(directorio, archivo)
        thresh_, contorno=preprocesar_imagen(ruta, False, 1)
        HuM=calcular_parametros_y_hu(thresh_, contorno)
        if thresh_ is not None:
            imagenes.append(HuM)
            etiquetas.append(etiqueta)
    return imagenes, etiquetas

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
        _, imagen_binaria = cv2.threshold(imagen_gauss, 150, 255, cv2.THRESH_BINARY)

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
            elongacion = a / b 
        else:
            elongacion = b / a
    else:
        print("No hay suficientes puntos para la elongación")
        elongacion = None  # No se puede calcular la elongación si no hay suficientes puntos
    
    # Crear el vector de características
    if elongacion is not None:
        vector_caracteristicas = np.append(momentosHu, [relacion_aspecto, circularidad, elongacion])
    else:
        vector_caracteristicas = np.append(momentosHu, [relacion_aspecto, circularidad])
    
    return vector_caracteristicas

def normalizar(datos, maximo, minimo):   
    # Obtener los valores mínimos y máximos de cada característica en X
    return (datos-minimo)/(maximo-minimo)

# Directorio que contiene las imágenes de entrenamiento de tuercas
directorio_tuercas = "Tuercas entrenamiento/"
tuercas_entrenamiento, etiquetas_tuercas_entrenamiento = cargar_imagenes_y_etiquetas(directorio_tuercas, etiqueta=0)

# Directorio que contiene las imágenes de entrenamiento de tornillos
directorio_tornillos = "Tornillos entrenamiento/"
tornillos_entrenamiento, etiquetas_tornillos_entrenamiento = cargar_imagenes_y_etiquetas(directorio_tornillos, etiqueta=1)

# Directorio que contiene las imágenes de entrenamiento de clavos
directorio_clavos = "Clavos entrenamiento/"
clavos_entrenamiento, etiquetas_clavos_entrenamiento = cargar_imagenes_y_etiquetas(directorio_clavos, etiqueta=2)

# Directorio que contiene las imágenes de entrenamiento de arandelas
directorio_arandelas = "Arandelas entrenamiento/"
arandelas_entrenamiento, etiquetas_arandelas_entrenamiento = cargar_imagenes_y_etiquetas(directorio_arandelas, etiqueta=3)

# Directorio que contiene las imágenes de validación de tuercas
directorio_tuercas = "Tuercas validacion/"
tuercas_validacion, etiquetas_tuercas_validacion = cargar_imagenes_y_etiquetas(directorio_tuercas, etiqueta=0)

# Directorio que contiene las imágenes de validación de tornillos
directorio_tornillos = "Tornillos validacion/"
tornillos_validacion, etiquetas_tornillos_validacion = cargar_imagenes_y_etiquetas(directorio_tornillos, etiqueta=1)

# Directorio que contiene las imágenes de validación de clavos
directorio_clavos = "Clavos validacion/"
clavos_validacion, etiquetas_clavos_validacion = cargar_imagenes_y_etiquetas(directorio_clavos, etiqueta=2)

# Directorio que contiene las imágenes de validación de arandelas
directorio_arandelas = "Arandelas validacion/"
arandelas_validacion, etiquetas_arandelas_validacion = cargar_imagenes_y_etiquetas(directorio_arandelas, etiqueta=3)

# Directorio que contiene las imágenes de prueba de tuercas
directorio_tuercas = "Tuercas prueba/"
tuercas_prueba, etiquetas_tuercas_prueba = cargar_imagenes_y_etiquetas(directorio_tuercas, etiqueta=0)

# Directorio que contiene las imágenes de prueba de tornillos
directorio_tornillos = "Tornillos prueba/"
tornillos_prueba, etiquetas_tornillos_prueba = cargar_imagenes_y_etiquetas(directorio_tornillos, etiqueta=1)

# Directorio que contiene las imágenes de prueba de clavos
directorio_clavos = "Clavos prueba/"
clavos_prueba, etiquetas_clavos_prueba = cargar_imagenes_y_etiquetas(directorio_clavos, etiqueta=2)

# Directorio que contiene las imágenes de prueba de arandelas
arandelas_prueba, etiquetas_arandelas_prueba = cargar_imagenes_y_etiquetas(directorio_arandelas, etiqueta=3)

# Concatenar todas las imágenes y etiquetas
x_entrenamiento = []
y_entrenamiento = []
x_validacion = []
y_validacion = []
x_prueba = []
y_prueba = []

x_entrenamiento = np.concatenate((tuercas_entrenamiento, tornillos_entrenamiento, clavos_entrenamiento, arandelas_entrenamiento), axis=0)
y_entrenamiento = np.concatenate((etiquetas_tuercas_entrenamiento, etiquetas_tornillos_entrenamiento, etiquetas_clavos_entrenamiento, etiquetas_arandelas_entrenamiento))

x_validacion = np.concatenate((tuercas_validacion, tornillos_validacion, clavos_validacion, arandelas_validacion), axis=0)
y_validacion = np.concatenate((etiquetas_tuercas_validacion, etiquetas_tornillos_validacion, etiquetas_clavos_validacion, etiquetas_arandelas_validacion))

x_prueba = np.concatenate((tuercas_prueba, tornillos_prueba, clavos_prueba, arandelas_prueba), axis=0)
y_prueba = np.concatenate((etiquetas_tuercas_prueba, etiquetas_tornillos_prueba, etiquetas_clavos_prueba, etiquetas_arandelas_prueba))

# Entrenar el modelo Knn con los momentos de Hu seleccionados
modelo_seleccionado = Knn(k=4)
modelo_seleccionado.fit(x_entrenamiento, y_entrenamiento)

# Ajustar hiperparámetros usando el conjunto de validación
mejor_precision_seleccionada = 0
mejor_k_seleccionado = None
for k in range(1, 10):
    modelo_seleccionado.k = k
    precision_validacion = modelo_seleccionado.score(x_validacion, y_validacion)
    if precision_validacion > mejor_precision_seleccionada:
        mejor_precision_seleccionada = precision_validacion
        mejor_k_seleccionado = k
print("Mejor valor de k encontrado:", mejor_k_seleccionado)
print("Precisión en el conjunto de validación:", mejor_precision_seleccionada)

# Entrenar el modelo Knn con el mejor valor de k y los momentos de Hu seleccionados
modelo_seleccionado.k = mejor_k_seleccionado
x_entrenamiento_total = np.concatenate((x_entrenamiento, x_validacion))
y_entrenamiento_total = np.concatenate((y_entrenamiento, y_validacion))
indices_seleccionados=[8, 1, 5, 9]
x_entrenamiento_total = x_entrenamiento_total[:,indices_seleccionados]
X_max=np.max(x_entrenamiento_total, axis=0)
X_min=np.min(x_entrenamiento_total, axis=0)
x_entrenamiento_total = normalizar(x_entrenamiento_total, X_max, X_min)
x_prueba = x_prueba[:,indices_seleccionados]
x_prueba = normalizar(x_prueba, X_max, X_min)
modelo_seleccionado.fit(x_entrenamiento_total, y_entrenamiento_total)

# Hacer predicciones en los datos de prueba
predicciones_seleccionadas = modelo_seleccionado.predecir(x_prueba)

# Cargar la imagen a clasificar
imagen_path="lapicera.jpg"
thresh_,contorno= preprocesar_imagen(imagen_path, True, 3)
dato_analizar=calcular_parametros_y_hu(thresh_,contorno)
dato_analizar = dato_analizar[indices_seleccionados]
dato_analizar = normalizar(dato_analizar, X_max, X_min)

# Realizar la predicción usando el modelo KNN con los momentos de Hu seleccionados
etiqueta_predicha_seleccionada = modelo_seleccionado.predecir(np.array([dato_analizar]))[0]

# Definir un diccionario para mapear las etiquetas a los nombres de las categorías
categorias = {
    0: "Tuerca",
    1: "Tornillo",
    2: "Clavo",
    3: "Arandela"
}

# Imprimir la etiqueta predicha
print("La imagen es:", categorias[etiqueta_predicha_seleccionada])

# Medida de referencia
referencia_path="referencia1.jpg"
referencia_original, referencia_filtrada = preprocesar_imagen(referencia_path, False, 2)
referencia_original = cv2.resize(referencia_original, (300, 250))
referencia_filtrada = cv2.resize(referencia_filtrada, (300, 250))
imagen_original, imagen_filtrada = preprocesar_imagen(imagen_path, False, 4)
# Para medir correctamente la imagen
referencia_original = cv2.resize(referencia_original, (300, 250))
referencia_filtrada = cv2.resize(referencia_filtrada, (300, 250))
imagen_original = cv2.resize(imagen_original, (300, 250))
imagen_filtrada = cv2.resize(imagen_filtrada, (300, 250))

# Calcular medida de clavo o tornillo
if (etiqueta_predicha_seleccionada==1 or etiqueta_predicha_seleccionada==2):
    height, width = medicion(referencia_original, referencia_filtrada, imagen_original, imagen_filtrada)
    if height>width:
        medida = height
    else:
        medida = width
    print("El ", categorias[etiqueta_predicha_seleccionada], "mide: ", medida, " cm.")

# Evaluar el modelo con los momentos de Hu seleccionados
precision_seleccionada = modelo_seleccionado.score(x_prueba, y_prueba)
print("La precisión del modelo con momentos de Hu seleccionados es:", precision_seleccionada)

# Calcular matriz de confusión
matriz_confusion = np.zeros((4, 4), dtype=int)
for y_verdadero, y_predicho in zip(y_prueba, predicciones_seleccionadas):
    matriz_confusion[y_verdadero, y_predicho] += 1

# Graficar matriz de confusión
graficar_matriz_confusion(matriz_confusion)

# Graficar los datos de entrenamiento
plot_data(x_entrenamiento_total, y_entrenamiento_total, categorias, dato_analizar)