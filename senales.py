import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Parámetros del vector de tiempo
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.arange(-2, 2, 1/fs)  # Vector de tiempo de -2 a 2 segundos

# frecuencia fundamental para las senales periodicas
frec = 2  # Hz

# creacion del conjunto de senales periodicas
senoidal = np.sin(2 * np.pi * frec * t)
cuadrada = signal.square(2 * np.pi * frec * t)
triangular = signal.sawtooth(2 * np.pi * frec * t, width=0.5)
diente_sierra = signal.sawtooth(2 * np.pi * frec * t, width=1)

# Visualización rápida para comprobar los arreglos
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
axs[0].plot(t, senoidal)
axs[0].set_title('Senoidal')
axs[1].plot(t, cuadrada)
axs[1].set_title('Cuadrada')
axs[2].plot(t, triangular)
axs[2].set_title('Triangular')
axs[3].plot(t, diente_sierra)
axs[3].set_title('Diente de Sierra')

for ax in axs:
    ax.grid(True)
    ax.set_xlim(-1, 1) 

plt.tight_layout()

#plt.show() #mostrar senales periodicas

