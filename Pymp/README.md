# Efecto-blur-con-Pymp
## Aplicar efecto blur a imagenes usando pymp que es la interza de OpenMp para python.

## Librerias
Usar Python3.*


[instalar opencv](https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)

Instalar pymp

```python
pip3 install pymp-pypi
```


Importar librerias


```python
import cv2
import pymp



```

## Ejecutar
Comando de ejecución:


```
python3 blur_OpenMp.py -i 720p.jpg -bx 3 -by 3 -t 3
```
Donde:


-i: Seleccionan la imagen a tratar


-bx: Cantidad de kernel en el eje X


-by: Cantidad de kernel en el eje Y


-t: Numero de hilos




## Efecto blur 

El efecto blur es sumar los pixeles vecinos de un pixel y hayar el premedio incluyendolo y asignarle ese valor al pixel, asi toma un efecto borroso.

El trabajo que toma cada hilo es la imagen dividia en filas, la cantidad de filas a tratar son la cantidad de hilos. Es decir si por ejemplo son 4 hilos, el primer hilo toma hasta la fila 1/4, el segundo desde 1/4 hasta 2/4, y asi sucesivamente.


