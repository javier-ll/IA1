    def calcular_momentos_representativos(self):
        momentos_centroides = self.centroides
        media_momentos = np.mean(momentos_centroides, axis=0)
        desviacion_momentos = np.std(momentos_centroides, axis=0)

        # Ordenar las desviaciones estándar y seleccionar los índices de los tres momentos más representativos
        momentos_representativos_indices = np.argsort(desviacion_momentos)[:3]
        momentos_representativos = media_momentos[momentos_representativos_indices]

        return momentos_representativos_indices, momentos_representativos