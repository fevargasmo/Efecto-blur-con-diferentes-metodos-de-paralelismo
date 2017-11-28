import cv2
import sys
import pymp


def bluring(Imagen, PY, PX, BY, BX, hilo, imagenes):
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
    imagenes[hilo] = Imagen[PY[0]: PY[1]]
    # print("El hilo %s ha finalizado." % hilo)


def blur(Imagen, LongitudBuffer, NumeroHilos):
    BX = LongitudBuffer[0]
    BY = LongitudBuffer[1]
    py = len(Imagen)
    PX = len(Imagen[0])
    particion = 1 / NumeroHilos
    tp = int(py * particion)
    imagenes = pymp.shared.list(range(NumeroHilos))
    with pymp.Parallel(NumeroHilos) as hilo:
        PY = [0, 0]
        if hilo.thread_num == NumeroHilos:
            PY = [hilo.thread_num * tp, py, py]
        else:
            PY = [hilo.thread_num * tp, (hilo.thread_num + 1) * tp, py]
        bluring(Imagen, PY, PX, BY, BX, hilo.thread_num, imagenes)
    for hilo in range(NumeroHilos):
        Imagen[hilo * tp: (hilo + 1) * tp] = imagenes[hilo]

    # print('Todos los Hilos han finalizado')
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
