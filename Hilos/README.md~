# Efecto-blur-con-Hilos-Python
Aplicar efecto blur a imagenes usando hilos de python.

## Librerias
Se necesita importar la libreria threading de python e [instalar opencv 3](https://linuxhint.com/how-to-install-opencv-on-ubuntu/) 
```python
import cv2
import threading
```

## Ejecutar
Para ejecutar el codigo usar esta linea:
```
python blur_Threads_PY.py 720p.jpg 3 3 2
```
python blur_Threads_PY.py -i -bx -by -t


Donde:


-i: imagen a tratar.


-bx: kernel en x.


-by: kernel en y.


-t: numero de hilos.



## Efecto blur 

El efecto blur es sumar los pixeles vecinos de un pixel y hayar el premedio incluyendolo y asignarle ese valor al pixel, asi toma un efecto borroso.

El trabajo que toma cada hilo es la imagen dividia en filas, la cantidad de filas a tratar son la cantidad de hilos. Es decir si por ejemplo son 4 hilos, el primer hilo toma hasta la fila 1/4, el segundo desde 1/4 hasta 2/4, y asi sucesivamente.



