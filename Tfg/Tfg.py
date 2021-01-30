#Inicio del programa


import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
from astropy.table import QTable
from mpl_toolkits import mplot3d


#El modelo




def resetear(R,L,T,x,y,M):
    global X
    global Y
    global Z
    global mu
    global Rt
    global Lt
    global Tc
    global Mt
    global fP
    global fT
    global fM
    global fL
    global Delta1_P
    global Delta2_P
    global Delta1_L
    global Delta2_L
    global Delta1_M
    global Delta1_T
    global Pest
    global Pcal
    global Test
    global Tcal
    global Mcal
    global Lcal
    global n
    global P_ajust
    global T_ajust
    global M_ajust
    global  L_ajust
    global ciclo
    global Fase
    global A1
    global A2
    global Cm
    global Cp
    global Ct
    global Ct_
    global Error
    global rho
    global X1
    global gen
    global Cm1
    global gen

    gen=np.zeros(101)

    Lt=L
    Tc=T
    Rt=R
    X = x
    Y = y
    Z = 1 - X - Y
    Mt = M
    mu = 1 / (2 * X + 0.75 * Y + 0.5 * Z)
    A1 = 1.9022 * mu * Mt
    A2 = 10.645 * math.sqrt(Mt / (mu * Z * (1 + X) * Lt))
    Cm = 0.01523 * mu
    Cm1=Cm*(10**8)/((10**5)*4*np.pi)
    Cp = 8.084 * mu
    Ct = 0.01679 * Z * (1 + X) * (mu ** 2)
    Ct_ = 3.234 * mu
    Error = 0.0001
    rho = np.zeros(100)
    X1 = X
    A2 = 10.645 * math.sqrt(Mt / (mu * Z * (1 + X) * Lt))
    fP = np.zeros(101)
    fT = np.zeros(101)
    fM = np.zeros(101)
    fL = np.zeros(101)
    Delta1_P = np.zeros(101)
    Delta2_P = np.zeros(101)
    Delta1_T = np.zeros(101)
    Delta1_M = np.zeros(101)
    Delta1_L = np.zeros(101)
    Delta2_L = np.zeros(101)
    Pest = np.zeros(101)
    Pcal = np.zeros(101)
    Test = np.zeros(101)
    Tcal = np.zeros(101)
    Mcal = np.zeros(101)
    Lcal = np.zeros(101)

    n = np.zeros(101)

    gen = np.zeros(101)

    P_ajust = np.zeros(2)
    T_ajust = np.zeros(2)
    M_ajust = np.zeros(2)
    L_ajust = np.zeros(2)

    ciclo=np.zeros(101,dtype=object)
    Fase =np.zeros(101,dtype=object)

#Se definen las funciones que se van a utilizar a lo largo del programa

#Paso 2: Calcula la presión y temperatura junto con sus derivadas y sus diferencias. Como valores de entrada utiliza
#valores de presión y temperatura de la capa anterior,paso de integración h y la capa i que se desea calcular

def paso2(P,T,h,i):
    Delta1_P[i-1]=h*(fP[i-1]-fP[i-2])
    Delta2_P[i-1]=h*(fP[i-1]-2*fP[i-2]+fP[i-3])
    Delta1_T[i-1]=h*(fT[i-1]-fT[i-2])
    Pest[i]=P[i-1]+h*fP[i-1]+0.5*Delta1_P[i-1]+5*Delta2_P[i-1]/12
    Test[i]=T[i-1]+h*fT[i-1]+0.5*Delta1_T[i-1]

#Paso3: Calcula la masa, su derivada y su diferencias. Como valores de entrada se necesita presion, temperatura, radio
#paso de integración h y la capa i que se quiere calcular

def paso3(P,T,r,h,i):
    fM[i]=Cm*P*(r[i]**2)/T
    Delta1_M[i]=h*(fM[i]-fM[i-1])
    Mcal[i]=Mcal[i-1]+h*fM[i]-0.5*Delta1_M[i]

#Paso4: Se recalcula la derivada de la presión con los valores estimados de temparatura y presión. Valores de entrada
#de la masa, radio, paso de integración y capa i

def paso4(M,r,h,i):
    fP[i]=-Cp*Pest[i]*M/(Test[i]*(r[i]**2))
    Delta1_P[i]=h*(fP[i]-fP[i-1])
    Pcal[i]=Pcal[i-1]+h*fP[i]-0.5*Delta1_P[i]

#Paso 7: Se recalcula la derivada de la temperatura con los valores estimados de temparatura y presión tomando
#

def paso7(L,r,h,i):
    fT[i]=-Ct*(Pcal[i]**2)*L/((Test[i]**8.5)*(r[i])**2)
    Delta1_T[i]=h*(fT[i]-fT[i-1])
    Tcal[i]=Tcal[i-1]+h*fT[i]-0.5*Delta1_T[i]

#Paso 6: Calcula la luminosidad y sus derivadas a partir de los parámetros de la generación de energía

def paso6(P,T,r,eps1,nu,X2,h,i):
    fL[i]=0.01845*eps1*X1*X2*(10**nu)*(mu**2)*(P[i]**2)*(T[i]**(nu-2))*(r[i]**2)
    Delta1_L[i]=h*(fL[i]-fL[i-1])
    Delta2_L[i]=h*(fL[i]-2*fL[i-1]+fL[i-2])
    Lcal[i]=Lcal[i-1]+h*fL[i]-0.5*Delta1_L[i]-(Delta2_L[i]/12)

#Paso 9: Se calcula el parámetro n+1 para estudiar cuando exista la convección

def paso9(P,T,i):
    return T*fP[i]/(P*fT[i])

#Error_rel: calcula error relativo entre dos magnitudes

def error_rel(a,b):
    return abs((a-b)/b)

#Paso2bis: Calcula la temperatura para la inntegraci

def paso2bis(h,i):
    Delta1_T[i-1]=h*(fT[i-1]-fT[i-2])
    Test[i]=Tcal[i-1]+h*fT[i-1]+0.5*Delta1_T[i-1]

#Paso polítropo: Se estima la presión a partir de la expresión del polítropo

def politropo(T,i,k):
    return k*(T[i]**2.5)

#Paso7bis: Se recalcula el gradiente de T a partir de la masa calculada

def paso7bis(r,h,i):
    fT[i]=-Ct_*Mcal[i]/(r[i]**2)
    Delta1_T[i]=h*(fT[i]-fT[i-1])
    Tcal[i]=Tcal[i-1]+h*fT[i]-0.5*Delta1_T[i]

#Funcion generacion de energía: Se toma la temperatura y la densidad como valores de entrada, evaluandola y obteniendo los valores de
#epsilon 1 y nu para el cálculo de la luminosidad

def gen_energia(T,rho):
    T*=10
    eps1_PP=0
    eps1_CN=0
    nu_PP=0
    nu_CN=0
    X2_pp=X
    X2_CN=Z/3
    if 4<T<=6:
        eps1_PP=10**-6.84
        nu_PP=6
    elif 6<T<=9.5:
        eps1_PP=10**-6.04
        nu_PP=5
    elif 9.5<T<=12:
        eps1_PP=10**-5.56
        nu_PP=4.5
    elif 12<T<=16.5:
        eps1_PP=10**-5.02
        nu_PP=4
    elif 16.5<T<=24:
        eps1_PP=10**-4.4
        nu_PP=3.5
    eps_pp=eps1_PP*X1*X2_pp*(T**nu_PP)*rho
    if 12<T<=16:
        eps1_CN=10**-22.2
        nu_CN=20
    elif 16<T<=22.5:
        eps1_CN=10**-19.8
        nu_CN=18
    elif 22.5<T<=27.5:
        eps1_CN=10**-17.1
        nu_CN=16
    elif 27.5<T<=36:
        eps1_CN=10**-15.6
        nu_CN=15
    elif 36<T<=50:
        eps1_CN=10**-12.5
        nu_CN=13
    eps_CN=eps1_CN*X1*X2_CN*(T**nu_CN)*rho
    if eps_pp>=eps_CN:
        return eps_pp,eps1_PP,nu_PP,X2_pp,"PP"
    else:
        return eps_CN,eps1_CN,nu_CN,X2_CN,"CN"


#inter_lineal: Interpola linealmente para encontrar el radio para el que el parámetro n+1=2.5

def inter_lineal(a,b,i):
    return (2.5-a[i])*(b[i+1]-b[i])/(a[i+1]-a[i]) + b[i]

#inter_lineal1: Interpola los valores para un radio dado

def inter_lineal1(a,b,c,i):
    return (a[i+1]-a[i])*(c-b[i])/(b[i+1]-b[i]) + a[i]

#Se calculan las primeras tres capas de la integración desde la superficie

def Primerascapas_sup(i,Rt):
    h =- 0.9 * Rt / 100
    r = np.linspace(0.9 * Rt, 0, 101)
    while i < 3:
        Fase[i]='Inicio'
        ciclo[i]='--'
        Tcal[i] = A1 * (1 / r[i] - 1 / Rt)
        Pcal[i] = A2 * (Tcal[i] ** 4.25)
        Mcal[i] = Mt
        Lcal[i] = Lt
        rho[i]=Cm1*Pcal[i]/Tcal[i]
        fP[i] = -Cp * Pcal[i] * Mt / (Tcal[i] * (r[i] ** 2))
        fT[i] = -Ct * (Pcal[i] ** 2) * Lt / ((Tcal[i] ** 8.5) * (r[i] ** 2))
        fM[i] = 0
        fL[i] = 0
        i += 1

    return i,r,Pcal,Tcal,Mcal,Lcal,h,rho


#Al haber iniciado la integración para 0.9*Rt se calculan las capas extra hasta alcanzar el radio total

def capas_extra(Rt):
    r = np.linspace(0.9 * Rt, 0, 101)
    h = 0.9 * Rt / 100
    v = int((Rt - 0.9 * Rt) / h)
    resetear(Rt, Lt, Tc, X, Y, Mt)
    i = 0
    r1 = np.zeros(v)
    while i < v:
        r1[i] = r1[i - 1] + h
        r1[0] = r[0] + h
        Fase[i] = '--'
        ciclo[i] = '--'
        gen[i]=0
        Tcal[i] = A1 * (1 / r1[i] - 1 / Rt)
        Pcal[i] = A2 * (Tcal[i] ** 4.25)
        Mcal[i] = Mt
        Lcal[i] = Lt
        rho[i] = Cm1 * Pcal[i] / Tcal[i]
        i = i + 1

    return r1, Pcal, Tcal, Mcal, Lcal,Fase,ciclo,rho,i,gen


#Algoritmo A.1.1: calcula todas las magnitudes utilizando el metodo predictor-corrector
#considerando la masa y la luminosidad constantes

def Algoritmo_A11(i,r,h):
    loop1 = True
    while loop1:
        paso2(Pcal, Tcal,h, i)
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                paso4(Mt,r,h, i)
                e = error_rel(Pcal[i], Pest[i])
                if e > Error:
                    Pest[i] = Pcal[i]
                else:
                    loop3 = False
            paso7(Lt,r,h, i)
            e = error_rel(Tcal[i], Test[i])
            if e > Error:
                Test[i] = Tcal[i]
            else:
                loop2 = False
        Mcal[i-1]=Mt
        Lcal[i-1]=Lt
        paso3(Pcal[i], Tcal[i],r,h, i)
        rho[i] = Cm1 * Pcal[i] / Tcal[i]
        e = error_rel(Mcal[i], Mt)
        Fase[i] = 'A.1.1'
        ciclo[i] = '--'
        if e < Error:
            i=i+1
        else:
            loop1 = False

    return i,r,Pcal,Tcal,Mcal,Lt,rho

#Algortimo A.1.2: Calcula todas las variables considerando unicamente la luminosidad constante.
#Toma valores de entrada de radio, paso de integración

def Algoritmo_A12(i,r,h):
    loop1 = True
    while loop1:
        paso2(Pcal, Tcal,h, i)
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                paso3(Pest[i], Test[i],r,h, i)
                paso4(Mcal[i],r,h, i)
                e = error_rel(Pcal[i], Pest[i])
                if e > Error:
                    Pest[i] = Pcal[i]
                else:
                    loop3 = False
            paso7(Lt,r,h, i)
            e = error_rel(Tcal[i], Test[i])
            if e > Error:
                Test[i] = Tcal[i]
            else:
                loop2 = False
        rho[i] = Cm1 * Pcal[i] / Tcal[i]
        eps = gen_energia(Tcal[i],rho[i])
        Lcal[i-1]=Lt
        paso6(Pcal,Tcal,r, eps[1], eps[2], eps[3],h, i)
        e = error_rel(Lcal[i], Lt)
        Fase[i] = 'A.1.2'
        ciclo[i] = '--'
        if e < Error:
            i=i+1
        else:
            loop1 = False
    return i,r,Pcal,Tcal,Mcal,Lt,rho

#Algoritmo A.1.3: Calcula las magnitudes sin considerar ninguna variable constante


def Algoritmo_A13(i,r,h):
    loop1 = True
    while loop1:
        paso2(Pcal, Tcal,h, i)
        loop2 = True
        while loop2:
            loop3 = True
            while loop3:
                paso3(Pest[i], Test[i],r,h, i)
                paso4(Mcal[i],r,h, i)
                e = error_rel(Pcal[i], Pest[i])
                if e > Error:
                    Pest[i] = Pcal[i]
                else:
                    loop3 = False
            rho[i] = Cm1 * Pcal[i] / Tcal[i-1]
            eps = gen_energia(Tcal[i-1],rho[i])
            paso6(Pcal, Test,r, eps[1], eps[2], eps[3],h, i)
            paso7(Lcal[i],r,h, i)
            e = error_rel(Tcal[i], Test[i])
            if e > Error:
                Test[i] = Tcal[i]
            else:
                loop2 = False
        n[i] = paso9(Pcal[i], Tcal[i], i)
        gen[i]=eps[0]
        Fase[i] = 'A.1.3'
        ciclo[i] = eps[4]
        if n[i] > 2.5:
            i = i + 1
        else:
            loop1 = False

    return i,r,Pcal,Tcal,Mcal,Lcal,n,gen,rho,Fase,ciclo

#Algoritmo A.2: Algoritmo que permite calcular las variables en el núcleo, utiliza
#el conjunto de ecuaciones correspondiente al comportamiento convectivo


def AlgoritmoA2(k,i):
    loop1 = True
    while loop1:
        paso2bis(i)
        loop2 = True
        while loop2:
            Pest[i] = politropo(Test, i)
            paso3(Pest[i], Test[i], i)
            paso7bis(i)
            e = error_rel(Tcal[i], Test[i])
            if e > Error:
                Test[i] = Tcal[i]
            else:
                loop2 = False
        Pcal[i] = politropo(Tcal, i)
        eps = gen_energia(Tcal[i])
        paso6(Pcal, Test, eps[1], eps[2], eps[3], i)
        if r[i+1] > 0:
            i = i + 1
        else:
            loop1 = False

    return i,r,Pcal,Tcal,Mcal,Lcal

#Algoritmo A.2_cent: posee la misma estructura que el algoritmo A.2 pero detiene su ejecución en la frontera
#entre nucleo y envoltura que ya conoceremos de antemano

def AlgoritmoA2_cent(r,k,h,i,j):
    loop1 = True
    while loop1:
        paso2bis(h,i)
        loop2 = True
        while loop2:
            Pest[i] = politropo(Test, i,k)
            paso3(Pest[i],Test[i],r,h,i)
            paso7bis(r,h,i)
            e = error_rel(Tcal[i], Test[i])
            if e > Error:
                Test[i] = Tcal[i]
            else:
                loop2 = False
        Pcal[i] = politropo(Tcal, i,k)
        rho[i] = Cm1 * Pcal[i] / Tcal[i]
        eps = gen_energia(Tcal[i-1],rho[i])
        paso6(Pcal, Test,r, eps[1], eps[2], eps[3],h, i)
        gen[i]=eps[0]
        ciclo[i]=eps[4]
        Fase[i] = "A.2"
        if i == j:
            loop1 = False
        else:
            i = i + 1

    return i,r,Pcal,Tcal,Mcal,Lcal,gen,rho,Fase,ciclo


#Función que calcula las variables para las 3 primeras capas iniciando desde el centro de la estrella

def Primerascapas_cent(k,i,Tc,Rt):
    h = 0.9 * Rt / 100
    r=np.linspace(0,0.9*Rt,101)
    while i < 3:
        Tcal[i] = Tc - (0.008207 * (mu ** 2) * k * (Tc ** 1.5) * (r[i] ** 2))
        Pcal[i] = k * (Tcal[i] ** 2.5)
        rho[i] = Cm1 * Pcal[i] / Tcal[i]
        eps = gen_energia(Tc,rho[i])
        Mcal[i] = 0.005077 * mu * k * (Tc ** 1.5) * (r[i] ** 3)
        Lcal[i] = 0.006150 * eps[1] * X1 * eps[3] * (10 ** eps[2]) * (mu ** 2) * (k ** 2) * (Tc ** (3 + eps[2])) * (
                    r[i] ** 3)
        fM[i] = Cm * k * (Tcal[i] ** 1.5) * (r[i] ** 2)
        fL[i] = 0.01845 * eps[1] * (k ** 2) * (Tcal[i] ** (3 + eps[2])) * (r[i] ** 2)
        fT[i] = -Ct_ * Mcal[2] / (r[2] ** 2)
        fT[1] = -Ct_ * Mcal[1] / (r[1] ** 2)
        fT[2] = -Ct_ * Mcal[2] / (r[2] ** 2)

        eps=gen_energia(Tcal[i],rho[i])
        Fase[i]="centro"
        ciclo[i]='--'
        i = i + 1

    return i,r,Pcal,Tcal,Mcal,Lcal,h,eps[0],rho


#Cálculo de los errores relativos en la frontera de cada magnitud junto con el error relativo total

def Error_total(P_ajust,T_ajust,L_ajust,M_ajust):
    e_P = error_rel(P_ajust[1], P_ajust[0])
    e_T = error_rel(T_ajust[1], T_ajust[0])
    e_L = error_rel(L_ajust[1], L_ajust[0])
    e_M = error_rel(M_ajust[1], M_ajust[0])
    return math.sqrt((e_P ** 2) + (e_T ** 2) + (e_M ** 2) + (e_L ** 2)),e_P,e_T,e_M,e_L

#Error_frontera: sirviendose de todos los algoritmos mencionados, calcula todas las variables para cada punto
#de la estrella. Realiza la interpolación lineal para la frontera calculando los errores relativos en ella.


def Error_frontera(Rt,Lt,Tc,X,Y,M):

    r, P, T, M, L, rho, gen, Fase, Ciclo, k, i, n = Integración_sup(Rt, Lt, Tc, X, Y, Mt)
    r_c, P_c, T_c, M_c, L_c, gen_c, rho_c, Fase_c, ciclo_c = Integrqacion_cent(Rt, Lt, Tc, X, Y, Mt, k, i)

    i=len(r)-2
    j=len(r_c)-2

    r_ajust = inter_lineal(n, r, i)

    P_ajust[0] = inter_lineal1(P, r, r_ajust, i)
    T_ajust[0] = inter_lineal1(T, r, r_ajust, i)
    M_ajust[0] = inter_lineal1(M, r, r_ajust, i)
    L_ajust[0] = inter_lineal1(L, r, r_ajust, i)

    P_ajust[1] = inter_lineal1(P_c, r_c, r_ajust, j)
    T_ajust[1] = inter_lineal1(T_c, r_c, r_ajust, j)
    M_ajust[1] = inter_lineal1(M_c, r_c, r_ajust, j)
    L_ajust[1] = inter_lineal1(L_c, r_c, r_ajust, j)

    Error,Error_P,Error_T,Error_M,Error_L=Error_total(P_ajust, T_ajust,L_ajust ,M_ajust)


    return Error,Error_P,Error_T,Error_M,Error_L,r_ajust,P_ajust,T_ajust,M_ajust,L_ajust


#Ajuste_T: realiza un ajuste de la temperatura central para un intervalo como variables de entrada, buscando el valor que minimiza el error
# y guardando toda la información en un vector para posteriormente representar gráficamente el error

def Ajuste_T(Rt,Lt,Tc_min,Tc_max,prec,X,Y,Mt):
    valor_minimo = Error_frontera(Rt, Lt, Tc,X,Y,Mt)[0]
    Tc_del_min=0
    f=int((Tc_max - Tc_min) / prec)+1
    Error=np.zeros(f+1)
    T=np.zeros(f+1)
    Tc_del_min = Tc_min
    i=0
    Error[0]=Error_frontera(Rt,Lt,Tc_min,X,Y,Mt)[0]
    while i<=f:
        T[i] = T[i-1] + prec
        T[0] = Tc_min
        Error[i]=Error_frontera(Rt,Lt,T[i],X,Y,Mt)[0]
        if Error[i]< valor_minimo:
            valor_minimo=Error[i]
            Tc_del_min=T[i]
            print(Tc_del_min)
        i = i + 1

    return valor_minimo,Tc_del_min,T,Error

#Ajuste total: ajusta los valores de radio y luminosidad para un intervalo dado, buscnado a su vez la temperatura
#central que minimiza el error para cada par de valores de radio y luminosidad. Toma como valores de entrada un intervalo de radios,
#luminosidades y temperaturas y el paso entre un valor y el siguiente. Devuelve el error junto con los valores iniciales
#que muestran el mínimo error

def Ajuste_total(Rt_min,Rt_max,Lt_min,Lt_max,Tc_min,Tc_max,prec_R,prec_L,prec_T,X,Y,Mt):
    i=0
    valor_minimo=Error_frontera(Rt_min,Lt_min,Tc_min,X,Y,Mt)[0]
    v=int((Rt_max-Rt_min)/prec_R)+1
    w=int((Lt_max-Lt_min)/prec_L)+1
    print(v,w)
    Error=np.zeros((v,w))
    T=np.zeros((v,w))
    R=np.zeros(v)
    L=np.zeros(w)
    Error[0,0]=Error_frontera(Rt_min,Lt_min,Tc,X,Y,Mt)[0]
    Rt_del_min=Rt_min
    Lt_del_min=Lt_min
    Tc_del_min=Tc_min
    while i<v:
        R[i]=R[i-1]+prec_R
        R[0] = Rt_min
        j=0
        while j<w:
            L[j]=L[j-1]+prec_L
            L[0] = Lt_min
            Ajuste=Ajuste_T(R[i],L[j],Tc_min,Tc_max,prec_T,X,Y,Mt)
            Error[i,j]=Ajuste[0]
            T[i,j]=Ajuste[1]
            print(R[i], L[j],Error[i,j],T[i,j])
            if Error[i,j]<valor_minimo:
                valor_minimo=Error[i,j]
                Rt_del_min=R[i]
                Lt_del_min=L[j]
                Tc_del_min=T[i,j]
                T1=Ajuste[2]
                Error1=Ajuste[3]
            j=j+1
        i=i+1
    return valor_minimo,Rt_del_min,Lt_del_min,Tc_del_min,R,L,Error,T1,Error1

#Ejecución: Se ejecuta el modelo completo y mostrando los resultados en una tabla.
#También se encarga de unir todas las soluciones en vectores únicos para su posterior representación gráfica

def Ejecucion(Rt,Lt,Tc,X,Y,Mt):

    rfinal, Pfinal, Tfinal, Mfinal, Lfinal, genfinal, rhofinal, Fasefinal, Ciclofinal, nfinal=Modelo_completo(Rt,Lt,Tc,X,Y,Mt)

    z=np.linspace(-11,0,12,dtype=int)
    z1=np.linspace(1,100,100,dtype=int)
    zfinal=np.hstack((z,z1))
    t=QTable([Ciclofinal,Fasefinal,zfinal,np.around(rfinal,decimals=5),np.around(Pfinal,decimals=7),np.around(Tfinal,decimals=7),np.around(Lfinal,decimals=6),np.around(Mfinal,decimals=6),np.around(nfinal,decimals=6)],names=('E','Fase','Capa','Radio','Presión','Temperatura','Luminosidad','Masa','n+1'))
    # t=QTable([Ciclofinal,Fasefinal,i,A3[1],Pfinal,Tfinal],names=('E','Fase','Capa','Radio','Presión','Temperatura'))
    t.pprint(max_lines=-1, max_width=-1)


    Error,Error_P,Error_T,Error_M,Error_L,r_ajust,P_ajust,T_ajust,M_ajust,L_ajust=Error_frontera(Rt, Lt, Tc,X,Y,Mt)
    print(Error_P)
    print("\nValores en la frontera")
    table = [[ "Superficie ",np.around(r_ajust,decimals=5), P_ajust[0], T_ajust[0], M_ajust[0], L_ajust[0]],
             [ "Centro ",np.around(r_ajust,decimals=5), P_ajust[1], T_ajust[1], M_ajust[1], L_ajust[1]],
             ["Error relativo (%)", " ",
              Error_P * 100,
              Error_T * 100,
              Error_M * 100,
              Error_L * 100]]
    print(tabulate(table, headers=["r", "P", "T", "M", "L"]))
    print("\nError relativo total en la frontera:      ", np.around(Error,decimals=5) * 100,'%')


#Gráficas Error: Se representan graficamente los errores relativos en la frontera tras haber realizado el ajuste
#de los valores iniciales.

def Graficas_Error(Rt_min,Rt_max,Lt_min,Lt_max,Tc_min,Tc_max,prec_R,prec_L,prec_T,X,Y,Mt):
    A=Ajuste_total(Rt_min,Rt_max,Lt_min,Lt_max,Tc_min,Tc_max,prec_R,prec_L,prec_T,X,Y,Mt)
    table = [["Error mínimo %", A[0] * 100], ["R que minimiza el error ", A[1]], ["L que minimiza el error ", A[2]],
             ["T que minimiza el error ", A[3]]]
    print(tabulate(table))
    ax=plt.axes()
    plt.plot(A[7],A[8])
    ax.set_xlabel('Temperatura')
    ax.set_ylabel('Error')
    plt.title('T que minimiza el error')
    fig = plt.figure()
    X,Y=np.meshgrid(A[4],A[5])
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, A[6], cmap = 'plasma', linewidth = 0, antialiased = True)
    plt.title('Error mínimo')
    ax.set_xlabel('Rt')
    ax.set_ylabel('Lt')
    ax.set_zlabel('Error')
    plt.show()

    return A[1],A[2],A[3]

#Gráficas variables: Realiza todas las representaciones gráficas de las soluciones generadas por la función
#modelo completo.Toma como valores de entrada radio, luminosidad total, temperatura central, composición
#químcia y masa total

def Graficas_variables(Rt,Lt,Tc,X,Y,Mt):

    r, P, T, M, L, gen, rho, Fase, Ciclo, nfinal=Modelo_completo(Rt,Lt,Tc,X,Y,Mt)


    plt.plot(r,gen,color='black')
    plt.title('Solucción para la generación de energía',fontsize=15)
    plt.xlabel('Radio de la estrella')
    plt.ylabel('Ritmo de generación de energía')
    plt.show()


    plt.plot(r,rho,color='magenta')
    plt.title('Solucción para la densidad',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$)',fontsize=15)
    plt.ylabel('Densidad (g/$cm^{3}$)',fontsize=15)
    plt.show()


    plt.plot(r,P)
    plt.title('Solución para la presión',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$)',fontsize=15)
    plt.ylabel('Presión ($10^{15}$ din $cm^{2}$)',fontsize=15)
    plt.show()

    plt.plot(r, T,color='orange')
    plt.title('Solución para la temperatura',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$)',fontsize=15)
    plt.ylabel('Temperatura ($10^{7}$ K)',fontsize=15)
    plt.show()

    plt.plot(r, L,color='red')
    plt.title('Solución para la luminosidad',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$)',fontsize=15)
    plt.ylabel('Luminosidad ($10^{33}$ erg/s)',fontsize=15)
    plt.show()

    plt.plot(r, M,color='green')
    plt.title('Solución para la masa',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$)',fontsize=15)
    plt.ylabel('Masa ($10^{33}$ g)',fontsize=15)
    plt.show()

    P=P/P[111]
    T=T/T[111]
    M=M/M[11]
    L=L/L[11]

    line1,=plt.plot(r, P)
    line2,=plt.plot(r, T)
    line3,=plt.plot(r, M)
    line4,=plt.plot(r, L)

    plt.xlabel('Radio de la estrella')
    plt.ylabel('Variables normalizadas')
    plt.title('Evolución en funcion del radio')
    plt.legend((line1,line2,line3,line4),('Presion','Temperatura','Masa','Luminosidad'))
    plt.show()


    plt.plot(r[0:108],Ciclo[0:108])
    plt.title('Ciclo responsable de la generación de energía',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Tipo de reacción nuclear',fontsize=15)
    plt.show()

    plt.title('Ritmo de energía en función de la reacción nuclear',fontsize=15)
    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    i=0
    while i<112:
        if Ciclo[i] == '--':
            Ciclo[i] = 0
        elif Ciclo[i]=='PP':
            Ciclo[i]=0.5
        elif Ciclo[i]=='CN':
            Ciclo[i]=1
        i=i+1


    plt.yticks([0,0.5,1],['--','PP','CNO'])

    line1,=plt.plot(r[0:110], gen[0:110]/gen[108])
    line2,=plt.plot(r[0:109], Ciclo[0:109])
    plt.title('Ritmo de generación de energía',fontsize=15)
    plt.xlabel('Radio ($10^{10}) cm$',fontsize=15)
    plt.legend((line1,line2),('Ritmo de generación de energía','Reacción nuclear'))

    plt.show()

#Variación X: Estudia como cambian los soluciones al variar X junto con sus representaciones gráficas.
#La función va variando la X, empezando en X_min hasta X_max con un paso entre un valor y el siguiente de
#prec_X. El e

def Variacion_X(X_min,X_max,prec_X,Y,Mt,Rt,Lt,Tc):
    v=int((X_max-X_min)/prec_X)+1
    X=np.zeros(v)
    P=[]
    T=[]
    L=[]
    M=[]
    energia=[]
    Fase=[]
    Ciclo=[]
    r=[]
    rho=[]
    e=np.zeros(v)
    x=np.zeros(v)
    Z=np.zeros(v)
    i=0
    while i<v:
        X[i]=X[i-1]+prec_X
        X[0] = X_min
        t=Ajuste_total(10,12.5,20,140,1.7,2.1,0.5,5,0.01,X[i],Y,Mt)
        E=Modelo_completo(t[1],t[2],t[3],X[i],Y,Mt)
        Fase.append(E[0])
        Ciclo.append(E[1])
        r.append(E[3])
        P.append(E[4])
        T.append(E[5])
        M.append(E[6])
        L.append(E[7])
        energia.append(E[8])
        rho.append(E[9])
        e[i]=Error
        Z[i]=np.around((1-X[i]-Y),decimals=3)
        i=i+1


    plt.xlabel('Composición de hidrógeno')
    plt.ylabel('Error')
    plt.title('Variación del error en función de X')
    plt.plot(X,e)
    plt.show()



    i = 0
    while i < len(P):
        plt.plot(r[i], P[i])
        i = i + 1


    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Presión ($10^{15}$ din $cm^{2}$)',fontsize=15)
    plt.title('Variación de la presión en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i = 0
    while i < len(rho):
        plt.plot(r[i], rho[i])
        i = i + 1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Densidad (g/$cm^{3}$)',fontsize=15)
    plt.title('Variación de la densidad en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()


    i=0
    while i<len(T):
        plt.plot(r[i],T[i])
        i=i+1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Temperatura ($10^{7}$ K)',fontsize=15)
    plt.title('Variación de la temperatura en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i=0
    while i<len(M):
        plt.plot(r[i],M[i])
        i=i+1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Masa ($10^{33}$ g)',fontsize=15)
    plt.title('Variación de la masa en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i=0
    while i<len(L):
        plt.plot(r[i],L[i])
        i=i+1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Luminosidad ($10^{33}$ erg/s)',fontsize=15)
    plt.title('Variación de la luminosidad en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i=0
    while i<len(Ciclo):
        plt.plot(r[i][0:110],Ciclo[i][0:110])
        i=i+1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Ciclo',fontsie=15)
    plt.title('Variación del ciclo en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i=0
    while i<len(Fase):
        plt.plot(r[i],Fase[i])
        i=i+1

    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Fase',fontsize=15)
    plt.title('Variación de la fase en función de X',fontsize=15)
    plt.legend((X[0], X[1], X[2]))
    plt.show()

    i=0
    while i<len(energia):
        plt.plot(r[i][0:110],energia[i][0:110])
        i=i+1


    f=('Z=',Z[0])
    plt.xlabel('Radio ($10^{10} cm$',fontsize=15)
    plt.ylabel('Gen de energía $erg$ $g^{-1}$ $ $s^{-1}$',fontsize=15 )
    plt.title('Variación de generación de energía en función de Z',fontsize=15)
    plt.legend(('Z=0.08','Z=0.05','Z=0.02'))
    plt.show()


    # return X,G,f[3],P,T,L,M

#Variación Y: Estudia como cambian los soluciones al variar Y junto con sus representaciones gráficas.
#Como valores de entradas toma la Y mínima con la que se quiere comenzar a probar

def Variacion_Y(Y_min,Y_max,prec_Y,X,Mt,Rt,Lt,Tc):
    v = int((Y_max - Y_min) / prec_Y) + 1
    Y = np.zeros(v)
    G = np.zeros(v)
    P = []
    T = []
    L = []
    M = []
    Ciclo = []
    Fase=[]
    energia=[]
    r=[]
    Z=np.zeros(v)
    i = 0
    while i < v:
        Y[i] = Y[i - 1] + prec_Y
        Y[0] = Y_min
        t = Ajuste_total(11, 13, 20, 140, 1.5, 2.1, 0.5, 5, 0.01, X, Y[i], Mt)
        f = Modelo_completo(t[1], t[2], t[3], X, Y[i], Mt)
        P.append(f[4])
        T.append(f[5])
        M.append(f[6])
        L.append(f[7])
        energia.append(f[8])
        Ciclo.append(f[1])
        Fase.append(f[0])
        r.append(f[3])
        Z[i]=np.around((1-X-Y[i]),decimals=3)
        i = i + 1

    i = 0
    while i < len(P):
        plt.plot(f[3], P[i])
        i = i + 1

    plt.xlabel('Radio ($10^{10} cm$', fontsize=15)
    plt.ylabel('Presión ($10^{15}$ din $cm^{2}$)',fontsize=15)
    plt.title('Variación de la presión en función de Y',fontsize=15)
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(T):
        plt.plot(r[i], T[i])
        i = i + 1

    plt.xlabel('Radio ($10^{10} cm$', fontsize=15)
    plt.ylabel('Temperatura ($10^{7}$ K)', fontsize=15)
    plt.title('Variación de la temperatura en función de Y',fontsize=15)
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(M):
        plt.plot(r[i], M[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Masa')
    plt.title('Variación de la masa en función de Y')
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(L):
        plt.plot(r[i], L[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Luminosidad')
    plt.title('Variación de la luminosidad en función de Y')
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(Ciclo):
        plt.plot(r[i][0:108], Ciclo[i][0:108])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Ciclo')
    plt.title('Variación del ciclo en función de Y')
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(Fase):
        plt.plot(r[i], Fase[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Fase')
    plt.title('Variación de la fase en función de Y')
    plt.legend((Y[0], Y[1], Y[2]))
    plt.show()

    i = 0
    while i < len(energia):
        plt.plot(r[i][0:110], energia[i][0:110])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Gen de energía')
    plt.title('Variación de generación de energía en función de Y')
    plt.legend((Z[0], Z[1], Z[2]))
    plt.show()

    # return Y, G, f[3], P, T, L, M

#Variación M: Estudia como cambian los soluciones al variar M junto con sus representaciones gráficas

def Variacion_M(M_min,M_max,prec_M,Rt,Lt,Tc,X,Y):
    v=int((M_max-M_min)/prec_M)+1
    M=np.zeros(v)
    G=np.zeros(v)
    P = []
    T = []
    L = []
    m=[]
    Fase=[]
    Ciclo=[]
    energia=[]
    r=[]
    rho=[]
    i=0
    while i<v:
        M[i]=M[i-1]+prec_M
        M[0]=M_min

        t = Ajuste_total(10, 13, 25, 100, 1.5, 2, 0.5, 5, 0.01, X, Y, M[i])
        f=Ejecucion(t[1],t[2],t[3],X,Y,M[i])
        P.append(f[4])
        T.append(f[5])
        L.append(f[7])
        m.append(f[6])
        energia.append(f[8])
        Fase.append(f[0])
        Ciclo.append(f[1])
        r.append(f[3])
        rho.append(f[9])
        i=i+1


    i = 0
    while i < len(P):
        plt.plot(r[i], P[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Presión')
    plt.title('Variación de la presión en función de M')
    plt.legend(( M[0], M[1], M[2]))
    plt.show()

    i = 0
    while i < len(P):
        plt.plot(r[i], rho[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Densidad')
    plt.title('Variación de la densidad en función de M')
    plt.legend((M[0], M[1], M[2]))
    plt.show()

    i = 0
    while i < len(T):
        plt.plot(r[i], T[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Temperatura')
    plt.title('Variación de la temperatura en función de M')
    plt.legend((M[0],M[1],M[2]))
    plt.show()

    i = 0
    while i < len(M):
        plt.plot(r[i], m[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Masa')
    plt.title('Variación de la masa en función de M')
    plt.legend((M[0],M[1],M[2]))
    plt.show()

    i = 0
    while i < len(L):
        plt.plot(r[i], L[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Luminosidad')
    plt.title('Variación de la luminosidad en función de M')
    plt.legend((M[0],M[1],M[2]))
    plt.show()

    i = 0
    while i < len(Ciclo):
        plt.plot(r[i][0:98], Ciclo[i][0:98])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Ciclo')
    plt.title('Variación del ciclo en función de M')
    plt.legend((M[0], M[1],M[2]))
    plt.show()

    i = 0
    while i < len(Fase):
        plt.plot(r[i], Fase[i])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Fase')
    plt.title('Variación de la fase en función de M')
    plt.legend((M[0],M[1],M[2]))
    plt.show()

    i = 0
    while i < len(energia):
        plt.plot(r[i][0:110], energia[i][0:110])
        i = i + 1

    plt.xlabel('Radio')
    plt.ylabel('Gen de energía')
    plt.title('Variación de generación de energía en función de M')
    plt.legend((M[0], M[1], M[2]))
    plt.show()

#Variación R: representa el error relativo  total en función del radio con el que se comienza la integración

def Variacion_R(Rt,Lt,Tc,X,Y,Mt):
    Rm=np.linspace(0.95*Rt,Rt/0.9,200)
    Error=np.zeros(200)
    nuevo=Modelo_Completo(Rm[0],Lt,Tc,X,Y,Mt)[0]
    Rmin=0
    i=0
    while i<200:
        Error[i]=Modelo_Completo(Rm[i],Lt,Tc,X,Y,Mt)[0]
        if Error[i]<nuevo:
            nuevo=Error[i]
            Rmin=Rm[i]*0.9
        i=i+1
    ax=plt.axes()
    ax.set_xlabel('Radio de la estrella')
    ax.set_ylabel('Error en la frontera')
    plt.title('Variación del radio inicial')
    Rm = np.linspace(10.008,11.79, 200)
    plt.plot(Rm,Error)
    plt.show()

    return nuevo,Rmin

#Integración sup:

def Integración_sup(Rt,Lt,Tc,X,Y,Mt):
    resetear(Rt, Lt, Tc, X, Y, Mt)
    i = 0
    i, r, Pcal, Tcal, Mcal, Lcal, h, rho = Primerascapas_sup(i, Rt)
    i, r, Pcal, Tcal, Mcal, Lcal, rho = Algoritmo_A11(i,r, h)
    i, r, Pcal, Tcal, Mcal, Lcal, rho = Algoritmo_A12(i, r, h)
    i, r, Pcal, Tcal, Mcal, Lcal, n, gen, rho, Fase, ciclo = Algoritmo_A13(i, r, h)

    r1, P, T, M, L, Fase1, ciclo1, rho1, z, gen1 = capas_extra(Rt)

    k = Pcal[i] / (Tcal[i] ** 2.5)

    r=np.hstack((r1[0:z][::-1],r[0:i+1]))
    n=np.hstack((np.zeros(z),n[0:i+1]))
    P=np.hstack((P[0:z][::-1],Pcal[0:i+1]))
    T=np.hstack((T[0:z][::-1],Tcal[0:i+1]))
    M=np.hstack((M[0:z][::-1],Mcal[0:i+1]))
    L=np.hstack((L[0:z][::-1],Lcal[0:i+1]))
    rho=np.hstack((rho1[0:z][::-1],rho[0:i+1]))
    gen=np.hstack((gen1[0:z][::-1],gen[0:i+1]))
    Fase=np.hstack((Fase1[0:z][::-1],Fase[0:i+1]))
    Ciclo=np.hstack((ciclo1[0:z][::-1],ciclo[0:i+1]))

    return r,P,T,M,L,rho,gen,Fase,Ciclo,k,i,n


def Integrqacion_cent(Rt,Lt,Tc,X,Y,Mt,k,i):
    b=0
    resetear(Rt,Lt,Tc,X,Y,Mt)
    j, r, Pcal, Tcal, Mcal, Lcal, h, eps, rho = Primerascapas_cent(k, b, Tc, Rt)
    j, r, Pcal, Tcal, Mcal, Lcal, gen, rho, Fase, ciclo = AlgoritmoA2_cent(r, k, h, j, 102 - i)

    return r[0:j],Pcal[0:j],Tcal[0:j],Mcal[0:j],Lcal[0:j],gen[0:j],rho[0:j],Fase[0:j],ciclo[0:j]

#Modelo completo: Une las soluciones de la integración desde la superficie y desde el centro en un único vector
#para su posterior representación gráfica. Toma como variables de entrada los parámetros constantes y los valores
#iniciales de radio, luminosidad y temperatura central.

def Modelo_completo(Rt,Lt,Tc,X,Y,Mt):

    r,P,T,M,L,rho,gen,Fase,Ciclo,k,i,n=Integración_sup(Rt,Lt,Tc,X,Y,Mt)
    r_c,P_c,T_c,M_c,L_c,gen_c,rho_c,Fase_c,ciclo_c=Integrqacion_cent(Rt,Lt,Tc,X,Y,Mt,k,i)

    rfinal=np.hstack((r[0:len(r)-1],r_c[0:len(r_c)-1][::-1]))
    Pfinal=np.hstack((P[0:len(P)-1],P_c[0:len(P_c)-1][::-1]))
    Tfinal=np.hstack((T[0:len(T)-1],T_c[0:len(T_c)-1][::-1]))
    Mfinal=np.hstack((M[0:len(M)-1],M_c[0:len(M_c)-1][::-1]))
    Lfinal=np.hstack((L[0:len(L)-1],L_c[0:len(L_c)-1][::-1]))
    genfinal=np.hstack((gen[0:len(gen)-1],gen_c[0:len(gen_c)-1][::-1]))
    rhofinal=np.hstack((rho[0:len(rho)-1],rho_c[0:len(rho_c)-1][::-1]))
    Fasefinal=np.hstack((Fase[0:len(Fase)-1],Fase_c[0:len(Fase_c)-1][::-1]))
    Ciclofinal=np.hstack((Ciclo[0:len(Ciclo)-1],ciclo_c[0:len(ciclo_c)-1][::-1]))
    nfinal=np.hstack((n[0:len(n)-1],np.zeros(len(r_c)-1)))


    return rfinal,Pfinal,Tfinal,Mfinal,Lfinal,genfinal,rhofinal,Fasefinal,Ciclofinal,nfinal



X=0.75
Y=0.2
Mt=5.1
Rt=11.793
Lt=50.23
Tc=1.8741

# print(Ajuste_total(11.785,11.795,50.1,50.3,1.874,1.875,0.0001,0.001,0.0001,0.75,0.2,5.1))

Rt=11.7949
Lt=50.299
Tc=1.8742

Variacion_X(0.72,0.78,0.03,Y,Mt,Rt,Lt,Tc)