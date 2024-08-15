import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_data(X, y, categorias, nuevo_punto=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for clase, etiqueta in categorias.items():
        ax.scatter(X[y == clase, 0], X[y == clase, 1], X[y == clase, 2], label=f'{etiqueta}')
    
    if nuevo_punto is not None:
        ax.scatter(nuevo_punto[0], nuevo_punto[1], nuevo_punto[2], color='r', s=100, label='Nuevo Punto', edgecolors='k')
    
    ax.set_xlabel('Circularidad')
    ax.set_ylabel('2do Momento de Hu')
    ax.set_zlabel('6to Momento de Hu')
    ax.set_title('Gráfica de dispersión de los datos en 3D')
    ax.legend()
    plt.show()

def graficar_matriz_confusion(matriz_confusion):
    plt.imshow(matriz_confusion, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(4), ["Tuercas", "Tornillos", "Clavos", "Arandelas"], rotation=45)
    plt.yticks(np.arange(4), ["Tuercas", "Tornillos", "Clavos", "Arandelas"])
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.title("Matriz de Confusión")
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(matriz_confusion[i, j]), horizontalalignment="center", color="black")
    plt.show()