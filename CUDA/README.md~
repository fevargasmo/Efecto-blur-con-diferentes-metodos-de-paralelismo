# Efecto blur con CUDA
## Aplicar efecto blur a imagenes usando CUDA

## Antes de empezar
*  [instalar CUDA 7.5](http://www.pradeepadiga.me/blog/2017/03/22/installing-cuda-toolkit-8-0-on-ubuntu-16-04/). Este link explica para cuda 8.0, pero entonces descargamos [cuda 7.5 de aca]https://developer.nvidia.com/cuda-75-downloads-archive) y simplemente cambiamos el archivo .deb

* [instalar opencv](https://linuxhint.com/how-to-install-opencv-on-ubuntu/) En este tutorial esta todo muy bien explicado


*Nota: Si no deja hacer el comando echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf, realizar:
```
sudo sh -c "echo '/usr/local/lib' >> /etc/ld.so.conf"
```


Este programa se realizo en Ubuntu 14.04 para evitar problemas de versionamiento con CUDA.
## compilar

```
nvcc -I/usr/include -L/usr/local/lib -g -o blur_CUDA blur_CUDA.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
```
*Nota: Para errores de librerias, depende como quedo intalado opencv, variar la linea -L/usr/local/lib por -L/usr/lib.
## Ejecutar
Comando de ejecución:


```
./blur-effect imagenIn.jpg imagenOut.jpg k t
```
Donde:


-imagenIn: Seleccionan la imagen a tratar


-imagenOut: nombre de la imagen resultante, se crea.


-k: Cantidad de kernel


-t: Numero de hilos




## Efecto blur 

El efecto blur es sumar los pixeles vecinos de un pixel y hayar el premedio incluyendolo y asignarle ese valor al pixel, asi toma un efecto borroso.

El trabajo que toma cada hilo es la imagen dividia en filas, la cantidad de filas a tratar son la cantidad de hilos. Es decir si por ejemplo son 4 hilos, el primer hilo toma hasta la fila 1/4, el segundo desde 1/4 hasta 2/4, y asi sucesivamente.
