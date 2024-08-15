#### Clasificador de tuercas, tornillos, clavos y arandelas utilizando visión artificial
El objetivo de este proyecto fue desarrollar un clasificador de imágenes para distinguir entre diferentes tipos
de objetos mecánicos: arandelas, tuercas, tornillos y clavos. Si el objeto detectado era un tornillo o clavo, se debía
obtener su medida real.
Se utilizaron los algoritmos KNN (K-Nearest Neighbors) y KMeans sin utilizar librerías especializadas, implementándolos 
desde cero en Python. Las imágenes fueron preprocesadas con filtros para extraer los momentos de Hu, 
que proporcionan características robustas de las formas de los objetos.
Los resultados demostraron una eficacia razonable en la clasificación de los objetos. Al utilizar la librería
OpenCV, se pudo calcular con mayor facilidad las medidas de los clavos y tornillos detectados. Los experimentos
concluyeron que la combinación de técnicas de preprocesamiento y clasificación sin librerías adicionales permite
un rendimiento adecuado, aunque la implementación de optimizaciones y el uso de herramientas especializadas
podrían mejorar aún más la precisión y eficiencia del sistema.
