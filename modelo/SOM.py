import numpy as np

class SOM:
    """ Implementación de la red Mapa Autoorganizado SOM """

    def __init__(self, n, m, eta, tau, sigma, sigma_dec):
        """ Inicialización de la clase SOM

        Args:
            n (int): Numero de filas de la red SOM
            m (int): Numero de columnas de la red SOM
            eta (float): Tasa de aprendizaje.
            tau (int): Factor de decreciiento de la tasa de aprendizaje.
            sigma (int): Radio de vecindad inicial.
            sigma_dec (int): Factor de decreciiento del radio de vecindad.
        """
        self.n = n
        self.m = m
        self.eta_inicial = eta
        self.eta = eta
        self.tau = tau
        self.sigma_inicial = sigma
        self.sigma = sigma
        self.sigma_dec = sigma_dec

        # Pesos sinapticos de la red
        self.pesos = None

    def obtener_nganadora(self, X):
        """ Funcion que obtiene la neurona ganadora

        Args:
            X (list): Datos de entrada

        Returns:
            int: indice en el que se encuentra la neurona ganadora
        """
        i_ganadora = np.array([0, 0])
        min_dist = np.iinfo(np.int64).max

        for x in range(self.pesos.shape[0]):
            for y in range(self.pesos.shape[1]):
                
                p = self.pesos[x, y, :]
                sq_dist = np.sum((p - X) ** 2)

                if sq_dist < min_dist:
                    min_dist = sq_dist
                    i_ganadora = np.array([x, y])
        
        return i_ganadora
    
    def entrenar(self, X, tam, max_iteraciones):
        """ funcion para entrenar la red SOM

        Args:
            X (list): Lista de entrada
            tam (tuple): Dimensión de los datos de entrada
            max_iteraciones (int): Numero maximo de iteraciones
        """

        self.pesos = np.random.rand(self.n, self.m, tam)

        for i in range(max_iteraciones):
            for e in X:
                # Se obtiene el índice de la neurona ganadora
                i_ganadora = self.obtener_nganadora(e)
            
                # Actualizar pesos
                for x in range(self.pesos.shape[0]):
                    for y in range(self.pesos.shape[1]):
                        p = self.pesos[x, y, :]
                        p_dist = np.sum((np.array([x, y]) - i_ganadora) ** 2)

                        if p_dist <= self.sigma ** 2:
                            influencia = np.exp(-p_dist / (2 * (self.sigma ** 2)))
                            nuevos_pesos = p + (self.eta * influencia * (e - p))
                            self.pesos[x, y, :] = nuevos_pesos

                # Decrecer vecindad
                factor = self.sigma_dec if i < 1000 else max_iteraciones
                time_constant = factor/np.log(self.sigma_inicial)
                self.sigma = self.sigma_inicial * np.exp(-i / time_constant)

                # Decrecer tasa de aprendizaje
                self.eta = self.eta_inicial * np.exp(-i / self.tau)
                if self.eta < 0.01:
                    self.eta = 0.01  

    def crear_mapa_dist(self):
        """ Funcion que crea un mapa con las distancias entre cada neurona.

        Returns:
            Matrix: Mapa de distancias entre neuronas
        """
        mapa = np.zeros((self.n, self.m))
        for x in range(self.n):
            for y in range(self.m):
                p = self.pesos[x, y, :]
                if x + 1 < self.n:
                    p1 = self.pesos[x+1, y, :]
                    mapa[x, y] += np.sqrt(np.sum((p - p1) ** 2))
                if y + 1 < self.m:
                    p1 = self.pesos[x, y+1, :]
                    mapa[x, y] += np.sqrt(np.sum((p - p1) ** 2))
        mapa /= mapa.max()
        return mapa