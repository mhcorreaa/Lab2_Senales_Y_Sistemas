import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# declarar path para guardar graficos en carpeta figuras

carpeta_figuras = Path("figuras")
carpeta_figuras.mkdir(exist_ok=True)

# senales periodicas 

fs = 1000  # frecuencia de muestreo (Hz)
t = np.arange(-2, 2, 1/fs)  # vector de tiempo de -2 a 2 segundos
frec = 2  

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

fig.savefig(
    carpeta_figuras / "señales_periodicas.pdf",
    bbox_inches="tight"
)


# senales aperiodicas 
    
u_t = np.heaviside(t, 1)
u_t_menos_1 = np.heaviside(t - 1, 1)
ventana = u_t - u_t_menos_1

exp_decreciente = np.exp(-t) * ventana
exp_creciente = np.exp(t) * ventana

impulso = np.zeros_like(t)
impulso[np.argmin(np.abs(t))] = 1

escalon = u_t

sinc_senal = np.zeros_like(t)
indices_no_cero = (t != 0)
sinc_senal[indices_no_cero] = np.sin(t[indices_no_cero]) / t[indices_no_cero]
sinc_senal[t == 0] = 1

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

fig2.savefig(
    carpeta_figuras / "senales_aperiodicas.pdf",
    bbox_inches="tight"
)

# funciones auxiliares para métricas y gráficos

dt = 1 / fs

def calcular_energia_potencia(senal, dt_val):
    energia = np.sum(np.abs(senal)**2) * dt_val
    duracion = len(senal) * dt_val
    potencia = energia / duracion if duracion > 0 else 0
    return energia, potencia

def visualizar_sistema(nombre_ent, senal_ent, nombre_ht, senal_ht, senal_sal, t_ent, t_ht, t_sal):
    E_ent, P_ent = calcular_energia_potencia(senal_ent, dt)
    E_ht, P_ht = calcular_energia_potencia(senal_ht, dt)
    E_sal, P_sal = calcular_energia_potencia(senal_sal, dt)

    fig_sys, axs_sys = plt.subplots(3, 1, figsize=(10, 8))
    fig_sys.suptitle(f'Sistema LTI: {nombre_ent} * {nombre_ht}', fontsize=12)

    axs_sys[0].plot(t_ent, senal_ent, color='blue')
    axs_sys[0].set_title(f'Entrada: {nombre_ent} | E={E_ent:.4f}, P={P_ent:.4f}')

    axs_sys[1].plot(t_ht, senal_ht, color='green')
    axs_sys[1].set_title(f'Respuesta al Impulso: {nombre_ht} | E={E_ht:.4f}, P={P_ht:.4f}')

    axs_sys[2].plot(t_sal, senal_sal, color='purple')
    axs_sys[2].set_title(f'Salida | E={E_sal:.4f}, P={P_sal:.4f}')

    for ax in axs_sys:
        ax.grid(True)

    plt.tight_layout()


# convolucion de senales periodicas y aperiodicas 

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

t_conv = np.linspace(2 * t[0], 2 * t[-1], len(senoidal) + len(impulso) - 1)
resultados_paso3 = {}

ejemplo_per = 'Cuadrada'
ejemplo_ap = 'Exp Decreciente'

for nombre_per, senal_per in senales_periodicas.items(): 
    for nombre_ap, senal_ap in respuestas_impulso.items():
        
        conv_resultado = np.convolve(senal_per, senal_ap, mode='full') * dt
        resultados_paso3[f"{nombre_per} * {nombre_ap}"] = conv_resultado
        
        
        if nombre_per == ejemplo_per and nombre_ap == ejemplo_ap:
            visualizar_sistema(nombre_per, senal_per, nombre_ap, senal_ap, conv_resultado, t, t, t_conv)

# factor de proporcionalidad y corrimiento temporal

factor = 5.0  
retraso_segundos = 1.0 
desplazamiento_muestras = int(retraso_segundos * fs)

respuestas_impulso_modificadas = {}

for nombre, senal in respuestas_impulso.items():
    senal_escalada = senal * factor 
    
    senal_desplazada = np.zeros_like(senal_escalada) 
    if desplazamiento_muestras > 0:
        senal_desplazada[desplazamiento_muestras:] = senal_escalada[:-desplazamiento_muestras]
    elif desplazamiento_muestras < 0:
        senal_desplazada[:desplazamiento_muestras] = senal_escalada[-desplazamiento_muestras:]
    else:
        senal_desplazada = senal_escalada
        
    respuestas_impulso_modificadas[f"{nombre} Modificada"] = senal_desplazada

senal_prueba = senales_periodicas['Cuadrada']
nombre_impulso_mod = 'Exp Decreciente Modificada'
impulso_mod = respuestas_impulso_modificadas[nombre_impulso_mod]

salida_modificada = np.convolve(senal_prueba, impulso_mod, mode='full') * dt

visualizar_sistema('Cuadrada', senal_prueba, nombre_impulso_mod, impulso_mod, salida_modificada, t, t, t_conv)

# convolucion entre senales aperiodicas

combinaciones_paso5 = [
    ('Impulso', 'Escalon'),
    ('Exp Decreciente', 'Escalon'),
    ('Exp Creciente', 'Exp Decreciente'),
    ('Sinc', 'Impulso'),
    ('Escalon', 'Sinc')
]

t_conv_ap = np.linspace(2 * t[0], 2 * t[-1], 2 * len(t) - 1)

for nombre_entrada, nombre_ht in combinaciones_paso5:
    senal_entrada = respuestas_impulso[nombre_entrada]
    senal_ht = respuestas_impulso[nombre_ht]
    
    conv_resultado = np.convolve(senal_entrada, senal_ht, mode='full') * dt
    
    visualizar_sistema(nombre_entrada, senal_entrada, nombre_ht, senal_ht, conv_resultado, t, t, t_conv_ap)

plt.show()