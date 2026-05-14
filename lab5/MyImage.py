import cv2
import matplotlib.pyplot as plt


class MyImage:
    def __init__(self, filename):
        self.original_img = cv2.imread(filename)
        self.processed_img = None
        self.last_filter = ''
    def show_image(self, version=0):
        if version != 0 and self.processed_img is not None:
            img = self.processed_img.copy()
            title = self.last_filter
        else:
            img = self.original_img.copy()
            title = 'Original Image'

        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()
        return

    def save_image(self, version=0):
        if version != 0 and self.processed_img is not None:
            img = self.processed_img.copy()
            title = self.last_filter
            title = title.replace(' ', '_')
        else:
            img = self.original_img.copy()
            title = 'Original_Image'

        filename = title+'.jpg'
        plt.imshow(img)
        plt.savefig(filename, dpi=200)
        plt.close()
        return

    def update_process_image(self, img, process_name=''):
        self.processed_img = img.copy()
        self.last_filter = process_name
        return