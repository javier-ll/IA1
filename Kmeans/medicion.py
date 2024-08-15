from scipy.spatial.distance import euclidean
from imutils import perspective
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

def mostrar_imagenes(referencia, objeto):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Imágenes medidas", fontsize=16)
    axs[0].imshow(referencia)
    axs[0].set_title('Imagen de referencia')
    axs[0].axis('off')
    axs[1].imshow(objeto)
    axs[1].set_title('Objeto a medir')
    axs[1].axis('off')

    plt.tight_layout()


    plt.show()

# Procesar imagen y encontrar contornos
def obtener_contornos(imagen):
    contorno = cv2.findContours(imagen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = imutils.grab_contours(contorno)
    contorno = [x for x in contorno if cv2.contourArea(x) > 100]
    return contorno

# Procesar la imagen del objeto de referencia
def medicion(referencia_original, referencia_filtrada, objeto_original, objeto_filtrado):
    ref_contornos = obtener_contornos(referencia_filtrada)
    # Se asume que el objeto de referencia tiene el contorno más grande de la imagen
    ref_objecto = max(ref_contornos, key=cv2.contourArea)
    ref_caja = cv2.minAreaRect(ref_objecto)
    ref_caja = cv2.boxPoints(ref_caja)
    ref_caja = np.array(ref_caja, dtype="int")
    ref_caja = perspective.order_points(ref_caja)
    (tl, tr, br, bl) = ref_caja
    dist_en_cm = 3.5
    dist_en_pixel = euclidean(tl, tr)
    pixel_por_cm = dist_en_pixel / dist_en_cm
    # Contorno más grande
    cv2.drawContours(referencia_original, [ref_caja.astype("int")], -1, (0, 0, 255), 2)
    punto_medio_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
    punto_medio_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
    ancho = euclidean(tl, tr) / pixel_por_cm
    alto = euclidean(tr, br) / pixel_por_cm
    cv2.putText(referencia_original, "{:.1f}cm".format(ancho), 
            (int(punto_medio_horizontal[0] - 15), int(punto_medio_horizontal[1] - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(referencia_original, "{:.1f}cm".format(alto), 
                (int(punto_medio_vertical[0] + 10), int(punto_medio_vertical[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Procesar imagen de objeto a medir
    objectos_contornos = obtener_contornos(objeto_filtrado)
    largest_cnt = max(objectos_contornos, key=cv2.contourArea)
    # Contorno más grande
    caja = cv2.minAreaRect(largest_cnt)
    caja = cv2.boxPoints(caja)
    caja = np.array(caja, dtype="int")
    caja = perspective.order_points(caja)
    (tl, tr, br, bl) = caja
    cv2.drawContours(objeto_original, [caja.astype("int")], -1, (0, 0, 255), 2)
    punto_medio_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
    punto_medio_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
    ancho = euclidean(tl, tr) / pixel_por_cm
    alto = euclidean(tr, br) / pixel_por_cm
    cv2.putText(objeto_original, "{:.1f}cm".format(ancho), 
            (int(punto_medio_horizontal[0] - 15), int(punto_medio_horizontal[1] - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(objeto_original, "{:.1f}cm".format(alto), 
                (int(punto_medio_vertical[0] + 10), int(punto_medio_vertical[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    mostrar_imagenes(referencia_original, objeto_original)

    return alto, ancho