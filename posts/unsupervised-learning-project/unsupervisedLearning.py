from matplotlib import pyplot as plt
import numpy as np
class UnsupervisedFunctions:
    

    def compare_images(self, A, A_):

        fig, axarr = plt.subplots(1, 2, figsize = (7, 3))

        axarr[0].imshow(A, cmap = "Greys")
        axarr[0].axis("off")
        axarr[0].set(title = "original image")

        axarr[1].imshow(A_, cmap = "Greys")
        axarr[1].axis("off")
        axarr[1].set(title = "reconstructed image")

    def svd_reconstruct(self, image, k):
    
        U, sigma, V = np.linalg.svd(image)

        D = np.zeros_like(image,dtype=float)
        D[:min(image.shape),:min(image.shape)] = np.diag(sigma)

        U_ = U[:,:k]
        D_ = D[:k, :k]
        V_ = V[:k, :]

        img_ = U_ @ D_ @ V_

        self.compare_images(image, img_)
    


