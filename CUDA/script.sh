#! /bin/bash

nvcc -I/usr/include -L/usr/local/lib -g -o blur_CUDA blur_CUDA.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

if [ -f times.txt ];
then
    rm times.txt
fi
image_720p="720p.jpg"
image_1080p="1080p.jpg"
image_4k="4k.jpg"
for image in $image_720p $image_1080p $image_4k;
do
    for size_kernel in 3 5 7 9 11 13 15;
    do
	echo "Times for image: "$image" kernel: "$size_kernel
	echo "Times for image: "$image" kernel: "$size_kernel >> times.txt
        for num_thread in 1 2 4 8 16;
	do
		output="$({ { /usr/bin/time -f '%e' ./blur_CUDA $image $size_kernel $num_thread; } 2>&1; })"
		echo $num_thread"  ""${output}" >> times.txt 
		#echo "${output}" >> times.txt
	done
    done
done
