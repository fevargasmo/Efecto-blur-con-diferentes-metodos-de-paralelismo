import cv2
import sys
import threading
import logging
import time


def bluring(Imagen, PY, PX, BY, BX):
    for y in range(PY[0], PY[1]):
        for x in range(PX):
            R, G, B, px = 0, 0, 0, 1
            for ky in range(-BY, BY + 1):
                for kx in range(-BX, BX + 1):
                    if y + ky >= 0 and y + ky + 1 <= PY[2] and x + kx >= 0 and x + kx + 1 <= PX:
                        R += int(Imagen[y + ky][x + kx][0])
                        G += int(Imagen[y + ky][x + kx][1])
                        B += int(Imagen[y + ky][x + kx][2])
                        px += 1
            Imagen[y][x][0] = R / px
            Imagen[y][x][1] = G / px
            Imagen[y][x][2] = B / px
    return Imagen


def blur(Imagen, LongitudBuffer, NumeroHilos):
    BX = LongitudBuffer[0]
    BY = LongitudBuffer[1]
    py = len(Imagen)
    PX = len(Imagen[0])
    particion = 1 / NumeroHilos
    tp = int(py * particion)
    for i in range(NumeroHilos):
        PY = [0, 0]
        if i + 1 == NumeroHilos:
            PY = [i * tp, py, py]
        else:
            PY = [i * tp, (i + 1) * tp, py]
        hilo = threading.Thread(target=bluring,args=(Imagen, PY, PX, BY, BX),name=i)
        hilo.start()

    while threading.active_count() > 1:
        continue
    return Imagen


def main(argv):
    if len(argv) > 1:
        bx, by = 3, 3
        h = 1
        debug = False
        for arg in range(len(argv)):
            if argv[arg] == "-i":
                Imagen = cv2.imread(argv[arg + 1])
            if argv[arg] == "-bx":
                bx = int(argv[arg + 1])
            if argv[arg] == "-by":
                by = int(argv[arg + 1])
            if argv[arg] == "-t":
                h = int(argv[arg + 1])
            if argv[arg] == "-d":
                debug = True
        Imagen = blur(Imagen, [bx, by], h)
        if(not debug):
            cv2.imshow('imagen', Imagen)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Uso: \n\t python blur.py -[argumento] [value] \n")
        print("\t Argumentos:\n\t\t")
        print("\tEjemplo: \n\t\t python blur.py - \n")


if __name__ == '__main__':
    main(sys.argv)
