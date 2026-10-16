import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# senales periodicas

fs = 1000  # frecuencia de muestreo (Hz)
t = np.arange(-2, 2, 1/fs)  # vector de tiempo de -2 a 2 segundos

# frecuencia fundamental para las senales periodicas
frec = 2  

# creacion del conjunto de senales periodicas
senoidal = np.sin(2 * np.pi * frec * t)
cuadrada = signal.square(2 * np.pi * frec * t)
triangular = signal.sawtooth(2 * np.pi * frec * t, width=0.5)
diente_sierra = signal.sawtooth(2 * np.pi * frec * t, width=1)

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

#plt.show() 


# senales aperiodicas
    
# define heaviside y desplazada
u_t = np.heaviside(t, 1)
u_t_menos_1 = np.heaviside(t - 1, 1)
ventana = u_t - u_t_menos_1

# exponencial decreciente y creciente restringidas al intervalo [0, 1)
exp_decreciente = np.exp(-t) * ventana
exp_creciente = np.exp(t) * ventana

# delta de kronecker
impulso = np.zeros_like(t)
impulso[np.argmin(np.abs(t))] = 1

# escalon unitario estandar
escalon = u_t

# funcion sinc definida como sen(t)/t manejando la division por cero
sinc_senal = np.zeros_like(t)
indices_no_cero = (t != 0)
sinc_senal[indices_no_cero] = np.sin(t[indices_no_cero]) / t[indices_no_cero]
sinc_senal[t == 0] = 1

# visualizacion 
fig2, axs2 = plt.subplots(5, 1, figsize=(10, 10))

axs2[0].plot(t, exp_decreciente)
axs2[0].set_title('Exponencial Decreciente')

axs2[1].plot(t, exp_creciente)
axs2[1].set_title('Exponencial Creciente')

axs2[2].plot(t, impulso)
axs2[2].set_title('Impulso')

axs2[3].plot(t, escalon)
axs2[3].set_title('Escalon')

axs2[4].plot(t, sinc_senal)
axs2[4].set_title('Sinc')

for ax in axs2:
    ax.grid(True)
    ax.set_xlim(-2, 2) 

plt.tight_layout()
#plt.show()

# convolucion de senales periodicas y aperiodicas

# se agrupan las senales 
senales_periodicas = {
    'Senoidal': senoidal,
    'Cuadrada': cuadrada,
    'Triangular': triangular,
    'Diente de Sierra': diente_sierra
}

respuestas_impulso = {
    'Exp Decreciente': exp_decreciente,
    'Exp Creciente': exp_creciente,
    'Impulso': impulso,
    'Escalon': escalon,
    'Sinc': sinc_senal
}

dt = 1 / fs
resultados_paso3 = {}

# El vector de tiempo de la salida abarca desde la suma de los tiempos iniciales 
# hasta la suma de los tiempos finales de los vectores de entrada
t_conv = np.linspace(2 * t[0], 2 * t[-1], len(senoidal) + len(impulso) - 1)

for nombre_per, senal_per in senales_periodicas.items():
    for nombre_ap, senal_ap in respuestas_impulso.items():
        # La convolución de señales discretas requiere el producto por dt para aproximar la integral
        conv_resultado = np.convolve(senal_per, senal_ap, mode='full') * dt
        resultados_paso3[f"{nombre_per} * {nombre_ap}"] = conv_resultado

# Graficamos un caso de prueba (Cuadrada con Exponencial Decreciente)
ejemplo_per = 'Cuadrada'
ejemplo_ap = 'Exp Decreciente'
ejemplo_conv = resultados_paso3[f"{ejemplo_per} * {ejemplo_ap}"]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].plot(t, senales_periodicas[ejemplo_per])
axs[0].set_title(f'Entrada: {ejemplo_per}')
axs[1].plot(t, respuestas_impulso[ejemplo_ap])
axs[1].set_title(f'Respuesta al Impulso: {ejemplo_ap}')
axs[2].plot(t_conv, ejemplo_conv)
axs[2].set_title('Salida (Convolución)')

for ax in axs:
    ax.grid(True)

plt.tight_layout()

# plt.show()