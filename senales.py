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

t_conv = np.linspace(2 * t[0], 2 * t[-1], len(senoidal) + len(impulso) - 1)

for nombre_per, senal_per in senales_periodicas.items(): 
    for nombre_ap, senal_ap in respuestas_impulso.items():
        
        conv_resultado = np.convolve(senal_per, senal_ap, mode='full') * dt    #convolucion
        resultados_paso3[f"{nombre_per} * {nombre_ap}"] = conv_resultado

# grafico caso de prueba (cuadrada con exponencial decreciente)
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


# factor de proporcionalidad y corrimiento temporal

factor = 5.0  # factor de proporcionalidad 
retraso_segundos = 1.0 
desplazamiento_muestras = int(retraso_segundos * fs)

respuestas_impulso_modificadas = {}

for nombre, senal in respuestas_impulso.items():
    
    senal_escalada = senal * factor    #producto
    
    senal_desplazada = np.zeros_like(senal_escalada)   # corrimiento temporal 
    if desplazamiento_muestras > 0:
        senal_desplazada[desplazamiento_muestras:] = senal_escalada[:-desplazamiento_muestras]
    elif desplazamiento_muestras < 0:
        senal_desplazada[:desplazamiento_muestras] = senal_escalada[-desplazamiento_muestras:]
    else:
        senal_desplazada = senal_escalada
        
    respuestas_impulso_modificadas[f"{nombre} Modificada"] = senal_desplazada

# comparacion
senal_prueba = senales_periodicas['Cuadrada']
impulso_original = respuestas_impulso['Exp Decreciente']
impulso_mod = respuestas_impulso_modificadas['Exp Decreciente Modificada']

salida_original = np.convolve(senal_prueba, impulso_original, mode='full') * dt
salida_modificada = np.convolve(senal_prueba, impulso_mod, mode='full') * dt


fig3, axs3 = plt.subplots(2, 1, figsize=(10, 6))

axs3[0].plot(t_conv, salida_original)
axs3[0].set_title('Salida Original (Cuadrada * Exp Decreciente)')

axs3[1].plot(t_conv, salida_modificada, color='orange')
axs3[1].set_title(f'Salida Modificada (Factor={factor}, Retraso={retraso_segundos}s)')

for ax in axs3:
    ax.grid(True)

plt.tight_layout()
#plt.show()


# convolucion entre senales aperiodicas 

combinaciones_paso5 = [
    ('Impulso', 'Escalon'),
    ('Exp Decreciente', 'Escalon'),
    ('Exp Creciente', 'Exp Decreciente'),
    ('Sinc', 'Impulso'),
    ('Escalon', 'Sinc')
]


t_conv_ap = np.linspace(2 * t[0], 2 * t[-1], 2 * len(t) - 1)

fig5, axs5 = plt.subplots(5, 1, figsize=(10, 12))

for i, (nombre_entrada, nombre_ht) in enumerate(combinaciones_paso5):
    senal_entrada = respuestas_impulso[nombre_entrada]
    senal_ht = respuestas_impulso[nombre_ht]
    
    conv_resultado = np.convolve(senal_entrada, senal_ht, mode='full') * dt
    
    axs5[i].plot(t_conv_ap, conv_resultado, color='purple')
    axs5[i].set_title(f'Salida | Entrada: {nombre_entrada} * H(t): {nombre_ht}')
    axs5[i].grid(True)

plt.tight_layout()

plt.show()