# crop images

def crop(img,k):
    img_height, img_width = img.shape[:2]
    step_height = img_height // k
    step_width = img_width // k
    img_cropped = []
    for j in range(0,img_width,step_width):
        for i in range(0,img_height,step_height):
            try:
                img_slice = img[i:i+step_height,j:j+step_width]
                img_cropped.append(img_slice)
            except:
                print('img_slice {},{} not possible to crop'.format(i,j))
    return img_cropped