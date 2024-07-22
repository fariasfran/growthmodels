a#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% LIBRERIAS Y CLASES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy #para ajustar sin modificar los datos originales
from directorios import *
#%
class dilat:

    def __init__(self, velocidad, temperatura, fraccion_ferrita):
        self.velocidad = velocidad
        self.temp = temperatura + 273 #PARA GRADOS CELCIUS QUITAR EL 273
        self.fracc_ferrita = fraccion_ferrita.rename('fracc_ferrita')

    def __repr__(self):
#        lista = [s.name for s in list(vars(self).values())[:]]
        lista = [s for s in vars(self).keys()]
        return '\n'.join(lista)

#############################
class dilatometria(dilat):

    def procesar(self):
        self.tasa = self.fracc_ferrita.diff() / self.temp.diff()
        self.tasa = self.tasa.rename('tasa')
        self.tasa_suave = self.tasa.rolling(3).mean().rename('tasa_suave')
        self.inversa_temp = 1/ self.temp.rename('inversa_temp')
        ###

    def agregar_pendientes(self, pendientes):
        #podria agregarse algo que mire si la longitud de datos_n es igual al
        #paso_ferrita y si no, mostrar un mensaje.
        self.pendientes = pd.Series(pendientes).rename('pendientes')

#############################
class discriminado(dilatometria):

    def __init__(self, velocidad, temperatura, fraccion_ferrita,
                 tasa, tasa_suave, paso_ferrita):
        '''
        Toma dos series de datos y reduce el paso según el valor indicado
        para la fracción de ferrita.
        '''
        fracciones_regulares = np.arange(paso_ferrita, 1, paso_ferrita)
        #hacemos una lista de indices
        indices = []
        for e in fracciones_regulares:
            #buscamos el indice del elemento mas cercano al valor e
            indice = (fraccion_ferrita - e).abs().argsort()[0]
            indices.append(indice)
        #Hacemos los datasets con los indices seleccionados
        datosX = []
        datosY = []
        datosTasa = []
        datosTasaSuave = []
        for e in indices:
            datosX.append(temperatura[e])
            datosTasa.append(tasa[e])
            datosTasaSuave.append(tasa_suave[e])
            datosY.append(fraccion_ferrita[e])
#        datosY = fracciones_regulares
        
        pd.Series(datosX)
        pd.Series(datosTasa)
        pd.Series(datosTasaSuave)
        pd.Series(datosY)

        self.velocidad = pd.Series(velocidad).rename('veloc_enfriamiento')
        self.temp = pd.Series(datosX).rename('temp')
        self.tasa = pd.Series(datosTasa).rename('tasa')
        self.tasa_suave = pd.Series(datosTasaSuave).rename('tasa_suave')
        self.fracc_ferrita = pd.Series(datosY).rename('fracc_ferrita')

#############################
class claferrita(dilat):
    
    def __init__(self, veloc_enfr, fracc_ferrita, tasa, tasa_suave, temp):
        self.veloc_enfriamiento = pd.Series(veloc_enfr).rename('veloc_enfriamiento')
        self.fracc_ferrita = pd.Series(fracc_ferrita).rename('fracc_ferrita')
        self.temp = pd.Series(temp).rename('temp')
#        self.tasa = self.fracc_ferrita.diff() / self.temp.diff()
        self.tasa = pd.Series(tasa).rename('tasa')
        self.tasa_suave = pd.Series(tasa_suave).rename('tasa_suave')


def buscar(b, e, vv=1):
    '''Toma un pd.Series, b, y un elemento a buscar, e. Devuelve el índice
    del elemento de b más cercanos a e.'''
    b = b[pd.notnull(b)] #por si hay elementos NaN
    indice = (b - e).abs().argsort().iloc[0]
    if vv: print('indice:\t', indice, '\nelem:\t', b.iloc[indice])
    return indice


#################
#################

#% CARGAMOS Y PROCESAMOS LOS DATOS NUEVOS.

claves = ['dp'+str(i)+'0' for i in [1,2,3,4,5,6,8]]

datap = { e : dilatometria( e,
    pd.read_csv(directorio + e + '.csv', sep=',')['temp'],
    pd.read_csv(directorio + e + '.csv', sep=',')['fracc_ferrita']    
        ) for e in claves }

datap['dp01'] = dilatometria( 'dp01',
    pd.read_csv(dir2, sep='\t')['temp gradC'],
    1 - pd.read_csv(dir2, sep='\t')['fracc aust']    
        )

datap['dp05'] = dilatometria( 'dp05', pd.read_csv(dir3, sep=',')['temp degC'],
                    1 - pd.read_csv(dir3, sep=',')['fracc aust'])

#%%

for k in datap: datap[k].procesar()

#% FUNCIONES

def funcion_SB(frac, tasa, n, m):
    pol = (1-frac)**n * frac**m
    return np.log(-tasa/pol)

def truncar(DS_truncar, DS_referencia, valorini = 0.1, valorfin = 0.9):
    '''Toma un par de datasets, el segundo debe ser fracción de ferrita, entre
    0 y 1; DS_truncar es uno cualquiera. Trunca ambos a valores de ff entre
    0.1 y 0.9.'''
    ind_ini = buscar(DS_referencia, valorini, vv=0)
    ind_fin = buscar(DS_referencia, valorfin, vv=0)
    return DS_truncar.iloc[ind_ini:ind_fin]

def lineavertical(x0, texto):
    '''Agrega una línea vertical al gráfico, además del texto indicado.
    Debe ejecutarse inmediatamente despues de creado el gráfico.'''
    ylim1, ylim2 = plt.axis()[2:] #leemos los valores de plt.ylim()
    plt.vlines(x0, ylim1, ylim2, colors='gray', lw=1)
    plt.text(1.001*x0, ylim1 + (ylim2-ylim1)*0.8, texto)

def guardarfig(directorio, nombre):
    '''Guarda la figura activa en el directorio, bajo el nombre.
    Formato .PNG, resolución 100dpi.'''
    guardarfig = directorio + nombre
    plt.savefig(guardarfig + '.png', format = 'png', dpi = 100)
    # plt.close()

def decorador():
    plt.title(ens)
    plt.xlabel('Inversa de temperatura, 1/K')
    plt.ylabel(r'$ln ( g(\alpha) / T^2 )$')
    plt.tight_layout()
    plt.show()

### FUNCIONES MODELO

def f4(alpha):
    '''Devuelve 4 *(1-alpha) *(-ln(1-alpha)) **3/4. alpha debe ser <1'''
    return 4 *(1-alpha) *(-np.log(1-alpha)) ** (3/4)

def f5(alpha):
    '''Devuelve 3 *(1-alpha) *(-ln(1-alpha)) **1/3. alpha debe ser <1'''
    return 3 *(1-alpha) *(-np.log(1-alpha)) ** 0.666666

def f6(alpha):
    '''Devuelve 2 *(1-alpha) *(-ln(1-alpha)) **1/2. alpha debe ser <1'''
    return 2 *(1-alpha) *(-np.log(1-alpha)) ** 0.5
# =============================================================================
def g1(alpha):
    '''Power law'''
    return alpha ** 0.25

def g3(alpha):
    '''Power law'''
    return alpha ** 0.5
# =============================================================================
def g4(alpha): #A4
    '''Avrami-Erofeev'''
    return (-np.log(1-alpha))**0.25

def g5(alpha): #A3
    '''Avrami-Erofeev'''
    return (-np.log(1-alpha))**0.333333

def g6(alpha): #A2
    '''Avrami-Erofeev'''
    return (-np.log(1-alpha))**0.5
# =============================================================================
def g7(alpha):
    '''One dimensional diffusion'''
    return alpha ** 2

def g8(alpha):
    '''Diffusion control (Janders)'''
    return ( 1 - (1-alpha) ** 0.333333 )**2

def g9(alpha):
    ''' Diffusion control (Crank)'''
    return 1.5 / ( 1/(1-alpha)**0.333333 -1 )

#%% Restringimos el estudio a 0.01:0.99 pct de ferrita

datos = {}
inicio, fin = 0.01, 0.99
#inicio, fin = 0.05 0.95

for ens in datap:
    frac = truncar(datap[ens].fracc_ferrita, datap[ens].fracc_ferrita,inicio, fin)
    tasa = truncar(datap[ens].tasa_suave, datap[ens].fracc_ferrita,inicio, fin)
    temp = truncar(datap[ens].temp, datap[ens].fracc_ferrita,inicio, fin)
    datos[ens] = (temp, frac, tasa)

#Borramos el de 5C/s
del(datos['dp05'])
# del(datos['dp01'])

#g5-A3 solido, g4-A4 punteada
colores = {
    'dp01' : 'gray',
    'dp05' : 'k',
    'dp10' : 'red',
    'dp20' : 'green',
    'dp30' : 'blue',
    'dp40' : 'c',
    'dp50' : 'm',
    'dp60' : 'orange',
    'dp80' : 'gold'
    }

plt.figure(figsize = (7, 4.55))
lista = ['dp01', 'dp10', 'dp20', 'dp30', 'dp40', 'dp50', 'dp60', 'dp80']
for e in lista:
    etiqueta = e[-2:] + r'$^\circ C/s$'
    plt.plot(datos[e][0] -273, datos[e][1], label = etiqueta, c = colores[e])
plt.legend()
plt.xlabel('Temperatura [C]')
plt.ylabel('Fracción de ferrita')
plt.tight_layout()
plt.show()

# guardarfig('../TESIS/IMAGENES/ISOCONVERSIONAL/', 'ff-T todos')

#%% DERIVA RESPECTO A 't' Y NO 'T'

# lista = ['dp01', 'dp10', 'dp20', 'dp30', 'dp40', 'dp50', 'dp60']
lista = ['dp01', 'dp10', 'dp20']
# lista = ['dp30', 'dp40', 'dp50']
# lista = ['dp20']

plt.figure(figsize = (6, 5))
for e in lista:
    x = datap[e].temp-273
    y = -datap[e].tasa_suave* int(e[-2:])
    etiqueta = e[-2:] + r'$^\circ$C/s'
    plt.plot(x, y, label = etiqueta, c = colores[e])

plt.legend()
plt.xlabel(r'Temperatura [$^\circ$C]')
plt.ylabel(r'Tasa de transformación, $\frac{d\alpha}{dt}$')
plt.ylim([-0.1, 1.1])
# plt.xlim([500, 820])
plt.xlim([600, 800])
plt.show()
plt.tight_layout()

# guardarfig('../TESIS/IMAGENES/ISOCONVERSIONAL/', 'tasa de transformacion 30-40-50')
# 30-40-50
# 01-10-20

#%% RANGO DE AJUSTE CON CRITERIO "diferencia = 0.1"

#g5-A3 solido, g4-A4 punteada
colores = {
    'dp05' : 'k',
    'dp10' : 'red',
    'dp20' : 'green',
    'dp30' : 'blue',
    'dp40' : 'c',
    'dp50' : 'm',
    'dp60' : 'orange',
    'dp80' : 'gold'
    }

rango_ajuste = {#indice inicial, indice final
    'dp01' : [ (None, 15), (86, None) ],
    'dp05' : [ (None, 5), (86, None) ], #simbolico, malos datos
    'dp10' : [ (None, 61), (86, None) ],
    'dp20' : [ (None, 77), (111, None) ],
    # 'dp30' : [ (None, 60), (132, None) ],
    'dp30' : [ (None, 86), (132, None) ],
    'dp40' : [ (None, 88), (138, 220) ],
    'dp50' : [ (None, 14), (61, None) ],
    'dp60' : [ (None, 14), (61, None) ],
    # 'dp80' : [ (1, 7), (61, None) ],
    'dp80' : [ (1, 10), (61, None) ],
    }

#%% CONSTRUCCION DE DATOS
ens = 'dp20'
modelo = g4 #ver cuáles
modelabel = str(modelo).split(' ')[1]

temp = deepcopy( datos[ens][0] )
frac = deepcopy( datos[ens][1] )
tasa = deepcopy( datos[ens][2] )
velocidad = int(ens[-2:])

y = np.log( modelo(frac) / temp**2 )

#%% GRAFICOS LOG VS 1/T
# plt.plot(1/temp, y, linestyle='dashed', c=colores[ens], label = ens)
# plt.plot(1/temp, y, linestyle='solid', c=colores[ens], label = ens)

plt.figure(figsize = (6,4))
# plt.figure(figsize = (5,4))

# ENSAYOS Y MODELOS
for m in [g4]: #agregar a la lista para más modelos
    # for ens in ['dp10', 'dp20', 'dp30', 'dp40', 'dp50']:
    for ens in ['dp'+str(e*10) for e in range(3,7)]:
        # if m == g4: estilo, etiqueta = 'dashed', None
        # else: estilo, etiqueta = 'solid', ens
        estilo, etiqueta = 'solid', ens #esto para un solo modelo
        
        temp = deepcopy( datos[ens][0] )
        frac = deepcopy( datos[ens][1] )
        tasa = deepcopy( datos[ens][1] )
        velocidad = int(ens[-2:])
        y = np.log( m(frac) / temp**2 )
        
        plt.plot(1/temp, y, linestyle = estilo, c=colores[ens], label = etiqueta)

decorador()
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0)) #notac cientif
plt.legend()
plt.title('')
plt.tight_layout()

plt.xlim([9.8e-4, 1.1e-3])
plt.ylim([-15, -13.25])

# guardarfig = '../TESIS/IMAGENES/ISOCONVERSIONAL/' + 'veloc altas modelo an'
# plt.savefig(guardarfig + '.svg', format = 'svg')

# =============================================================================
# for ff in [0.02, 0.1, 0.3, 0.6, 0.8, 0.9, 0.95]:
#     lineavertical(1/ temp.iloc[buscar(frac, ff, vv=0)], 'ff=' + str(ff))
# =============================================================================

#%% TASA DE TRANSFORMACIÓN

### DERIVADA RESPECTO A TEMPERATURA
plt.figure(figsize=(9,4))

plt.subplot(1, 2, 1)
plt.title('Derivada respecto a la temperatura')
plt.xlabel('Temperatura, K')
tasa = datos[ens][2]
temp = datos[ens][0] - 273
plt.plot(temp, -tasa)

### DERIVADA RESPECTO A TIEMPO
plt.subplot(1, 2, 2)
plt.title('Derivada respecto al tiempo (escala temp)')
plt.xlabel('Temperatura, K')
plt.plot(temp, -tasa * int(ens[-2:]) )
plt.scatter(temp, -tasa * int(ens[-2:]), c='k', s=3)

plt.tight_layout()
plt.show()

#%%## GRAFICO DE TASA Y MODELO - twinx()
ens = 'dp50'

temp = deepcopy( datos[ens][0] )
frac = deepcopy( datos[ens][1] )
tasa = deepcopy( datos[ens][2] )
velocidad = int(ens[-2:])

fig = plt.figure(1, figsize = (7,4))

ax1 = fig.add_subplot(111)
ax1.plot(1/temp, -tasa * int(ens[-2:]), c=colores[ens], lw=1, ls = '--', label = 'Tasa de transformación')
ax1.set_ylabel(r'Tasa de transformación, $\frac{d\alpha}{dt}$')
ax1.set_xlabel(r'Inversa de la temperatura [$1/K$]')
ax1.set_ylim([0,1.1])

ax2 = ax1.twinx()
ax2.set_title(ens[-2:] + r'$^\circ$C/s')
ax2.set_ylim([-15.5, -12.6])
# for modelo in [g4, g5, g6]:
#     modelabel = str(modelo).split(' ')[1]
#     y = np.log( modelo(frac) / temp**2 )
#     ax2.plot(temp, y, label = modelabel)
ax2.plot(1/temp, np.log(g4(frac) / temp**2), label = 'A4, 1/n = 0.25')
ax2.plot(1/temp, np.log(g5(frac) / temp**2), label = 'A3, 1/n = 0.33')
# ax2.plot(temp, np.log(g6(frac) / temp**2), label = 'A2, 1/n = 0.5')
ax2.set_ylabel(r'$ln \left( \frac{ g(\alpha)}{T^2} \right)$')

ax2.legend()
plt.tight_layout()
fig.show()

# guardarfig('../TESIS/IMAGENES/ISOCONVERSIONAL/', 'tasa y modelo an punteado vs 1-T '+ ens[-2:])

#%% TEMPORAL P CREAR TABLA
# for ens in ['dp'+str(_*10) for _ in range(1,7)]:
for ens in ['dp10']:
    for modelo in [g4, g5, g6]:
        runcell('CONSTRUCCION DE DATOS', '../DP/MODELO DIFUSION/CALCULOS/AjusteModelosCruzados 6 - ajuste con criterio.py')
        runcell('AJUSTAR RANGOS', '../DP/MODELO DIFUSION/CALCULOS/AjusteModelosCruzados 6 - ajuste con criterio.py')

#%% AJUSTAR RANGOS
a, b = rango_ajuste[ens][0]
x1 = 1/temp.iloc[a:b]
y1 = y.iloc[a:b]
a1, b1 = np.polyfit(x1, y1, 1)

plt.figure(figsize = (7,4))
plt.plot(1/temp, a1/temp + b1, c='k', label = 'Ajuste lineal')
plt.plot(1/temp, y, label = 'Puntos medidos', c=colores[ens])
# plt.scatter(1/temp, y, c='k', s=1)
plt.legend()
decorador()
plt.title(None)
# plt.title(ens + ' - modelo ' + modelabel)
print('\n')
print(f'{ens} ==> Energía {round(a1*8.314/1000,1)} kJ/mol \t Modelo {modelabel}')
print(f'{ens} ==> Pendiente {int(a1)} \t Modelo {modelabel}')
recta = np.poly1d([a1, b1])
print(f'{ens} ==> Fracc Ferr. fin {frac.iloc[buscar(y - recta(1/temp), -0.1, vv=0)]}')
print(f'{ens} ==> Temperatura fin {temp.iloc[buscar(y - recta(1/temp), -0.1, vv=0)]-273}')

# guardarfig('../TESIS/IMAGENES/ISOCONVERSIONAL/', 'diferencia')
# guardarfig = '../TESIS/IMAGENES/ISOCONVERSIONAL/' + 'diferencia'
# plt.savefig(guardarfig + '.svg', format = 'svg')

#%% BUSCAR INDICE DE RANGOS
buscar(y - recta(1/temp), -0.1)
1/temp.iloc[93]
temp.iloc[93] - 273
frac.iloc[93]

#%% INFORMACIÓN SOBRE RANGOS
for ens in ['dp01', 'dp10', 'dp20', 'dp30', 'dp40', 'dp50', 'dp60', 'dp80']:
    a,b = rango_ajuste[ens][0]
    temp = deepcopy( datos[ens][0] )
    frac = deepcopy( datos[ens][1] )
    tasa = deepcopy( datos[ens][1] )
    print('#########', ens, '#########')
    print('temp fin - temp inicio')
    print(f'{round(temp.iloc[a:b].iloc[-1])-273}-{round(temp.iloc[a:b].iloc[0])-273}')
    print('\nfracc inicio - fracc fin')
    print(f'{round(frac.iloc[a:b].iloc[0], 2)}-{round(frac.iloc[a:b].iloc[-1], 2)}')
    print('\n')


#%% CRITERIO LINEALIDAD
#Tomamos x1 y x2 de lo calculado anteriormente y armamos una recta

recta = np.poly1d([a1, b1])

#LOGARITMO Y RECTA DE AJUSTE
fig, [ax1, ax2] = plt.subplots(2, 1, figsize = (7,5))
ax1.set (title = ens)
ax1.plot(1/temp, a1/temp + b1, label = 'Ajuste lineal', c = 'k')
ax1.plot(1/temp, y, label = 'Puntos medidos', c = colores[ens])
ax1.set (ylabel = r'$ln ( g(\alpha) / T^2 )$')
# ax1.xaxis.set_visible(False)
ax1.legend()

#DIFERENCIA
ax2.plot(1/temp, (y - recta(1/temp)), c = 'k' )
ax2.set (ylabel = 'Diferencia')
ax2.set (xlabel = 'Inversa de temperatura, 1/K')
# ax2.xaxis.set_visible(False)

plt.tight_layout()
plt.show()

ax1.grid(axis = 'x')
ax2.grid()
# ax2.set_yticks([-0.1])

# plt.axhline(0, c = 'gray', lw = 1)

# ff=0.5
# lineavertical(1/ temp.iloc[buscar(frac, ff, vv=0)], 'ff=' + str(ff))




#%% GRAFICO DE ENERGÍAS
# velocidades = [1, 5] + [e*10 for e in range(1,7)] + [80]
# a4 = [462.1, 355.7, 197.1, 156.4, 181.3, 152.4, 213.6, 269.2, 345]
# a3 = [610.3, 468.5, 257.1, 202.9, 236.2, 197.4, 279.5, 353.7, 454.8]

velocidades = [1, 5] + [e*10 for e in range(1,7)]
a4 = [462.1, 355.7, 197.1, 156.4, 181.3, 152.4, 213.6, 269.2]
a3 = [610.3, 468.5, 257.1, 202.9, 236.2, 197.4, 279.5, 353.7]

plt.figure(figsize = (6, 5) )
for e, l in zip((a3, a4), ('Modelo A3, 1/n = 0.33', 'Modelo A4, 1/n = 0.25')):
    plt.scatter(velocidades, e, label = l)
    plt.plot(velocidades, e)

plt.ylim([100, 700])	

plt.legend()
plt.xlabel(r'Velocidad de enfriamiento $^\circ$C/s')
plt.ylabel('Energía de activación [kJ/mol]')
plt.grid(axis = 'y')
plt.xticks(velocidades)
plt.tight_layout()
plt.show()

# guardarfig('../TESIS/IMAGENES/ISOCONVERSIONAL/', 'energias de activacion')

#%%

###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
