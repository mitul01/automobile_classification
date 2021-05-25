import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class automobile:
    def __init__(self,filename):
        self.filename =filename

    def prediction(self):
        # load model
        model = load_model('model_new.h5')
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        print(result)
        if result[0][0] == 1:
            prediction = 'alto'
            return [{ "image" : prediction}]
        elif result[0][1]== 1:
            prediction = 'Honda 2020 CV-5'
            return [{ "image" : prediction}]
        elif result[0][2]== 1:
            prediction = 'i20'
            return [{"image": prediction}]


