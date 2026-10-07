import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. senales periodicas

# parametros del vector de tiempo
fs = 1000  # frecuencia de muestreo (Hz)
t = np.arange(-2, 2, 1/fs)  # vector de tiempo de -2 a 2 segundos

# frecuencia fundamental para las senales periodicas
frec = 2  

# creacion del conjunto de senales periodicas
senoidal = np.sin(2 * np.pi * frec * t)
cuadrada = signal.square(2 * np.pi * frec * t)
triangular = signal.sawtooth(2 * np.pi * frec * t, width=0.5)
diente_sierra = signal.sawtooth(2 * np.pi * frec * t, width=1)

# visualizacion rapida para comprobar los arreglos
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

# 2. senales aperiodicas
    
# se define la función escalon u(t) y la version desplazada u(t-1)
u_t = np.heaviside(t, 1)
u_t_menos_1 = np.heaviside(t - 1, 1)
ventana = u_t - u_t_menos_1

# exponencial decreciente y creciente restringidas al intervalo [0, 1)
exp_decreciente = np.exp(-t) * ventana
exp_creciente = np.exp(t) * ventana

# delta de dirac discreto
impulso = np.zeros_like(t)
impulso[np.argmin(np.abs(t))] = 1

# escalon unitario estandar
escalon = u_t

# funcion sinc definida como sen(t)/t manejando la division por cero
sinc_senal = np.zeros_like(t)
indices_no_cero = (t != 0)
sinc_senal[indices_no_cero] = np.sin(t[indices_no_cero]) / t[indices_no_cero]
sinc_senal[t == 0] = 1
