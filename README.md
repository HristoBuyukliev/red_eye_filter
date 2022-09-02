# Red eye filtering



## 1. How to run
```
cd python
pip install -r requirements.txt
python solution.py --input-images <input bin file>
```
### 1.1 optional arguments:
```
--threshold-red # by default 200
--correction-red # by default 150
--min-size
--max-size # these define what size eyes we are looking for. By default, we're only looking at eyes of size 1
```

## How the solution works:
I define four masks:

![masks](https://github.com/HristoBuyukliev/red_eye_filter/blob/master/masks.png?raw=true)

I apply the first one as a 2d convolutional filter along the image. If it's active we have an eye for sure there. Then, for each detection location, if any of the masks is active, I reduce the red channel by 150. 

## Speed hacks
1. I am only using the red channel, so I'm only working with it. This saves time in the conversion between numpy and `StridedImage`. 
2. I am using scipy's convolve2d instead of rolling out my own convolution. It's using numpy, which is using Fortran, so it's much faster than what Python would be. It's also vectorized. 
3. I'm only applying one mask, instead of four, and then detecting the four eyes only for the pixels which are a match. 

## Notes:
1. I am saving the cleared images as the same format bin files, which takes a lot of time. I might have also made a mistake in the format of the bin files somewhere.