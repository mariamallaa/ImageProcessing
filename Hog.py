from sklearn.datasets import fetch_lfw_people
from skimage import data, transform
from sklearn.feature_extraction.image import PatchExtractor
from itertools import chain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from commonfunctions import *
from skimage.color import rgb2gray
from skimage.transform import resize





def GetHogfeatures(img):
    hy=[[0,0,0],[-1,0,1],[0,0,0]]

    hx=[[0,-1,0],[0,0,0],[0,1,0]]

    gray_img = rgb2gray(img1)*255

    resized_image=resize(gray_img,(200,200))

    windowx=16
    windowy=16

    for i in range(0,resized_image.shape[0], windowx):
        for j in range(0,resized_image.shape[1], windowy):
            current_window=resized_image[i:i+windowx,j:j+windowy]
            current_windowY=












def learning():
    faces = fetch_lfw_people()
    positive_patches = faces.images
    positive_patches.shape
    imgs_to_use = ['camera'] #, 'text', 'coins', 'moon','page', 'clock', 'immunohistochemistry','chelsea', 'coffee', 'hubble_deep_field'
    images = [color.rgb2gray(getattr(data, name)())for name in imgs_to_use]


    def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
        extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
        extractor = PatchExtractor(patch_size=extracted_patch_size,
                                max_patches=N, random_state=0)
        patches = extractor.transform(img[np.newaxis])
        if scale != 1:
            patches = np.array([transform.resize(patch, patch_size)for patch in patches])
        return patches

    negative_patches = np.vstack([extract_patches(im, 1000, scale)for im in images for scale in [0.5, 1.0, 2.0]])
    negative_patches.shape

    

    X_train = np.array([GetHogfeatures(im)for im in chain(positive_patches,negative_patches)])

    y_train = np.zeros(X_train.shape[0])

    y_train[:positive_patches.shape[0]] = 1


    cross_val_score(GaussianNB(), X_train, y_train)

    grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
    grid.fit(X_train, y_train)
    grid.best_score_
    grid.best_params_


    model = grid.best_estimator_
    model.fit(X_train, y_train)

    test_image = skimage.data.astronaut()
    test_image = skimage.color.rgb2gray(test_image)
    test_image = skimage.transform.rescale(test_image, 0.5)
    test_image = test_image[:160, 40:180]

    plt.imshow(test_image, cmap='gray')
    plt.axis('off');

    def sliding_window(img, patch_size=positive_patches[0].shape,
                    istep=2, jstep=2, scale=1.0):
        Ni, Nj = (int(scale * s) for s in patch_size)
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Ni, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                yield (i, j), patch
                
    indices, patches = zip(*sliding_window(test_image))
    patches_hog = np.array([feature.hog(patch) for patch in patches])
    patches_hog.shape
    labels = model.predict(patches_hog)
    labels.sum()
    fig, ax = plt.subplots()
    ax.imshow(test_image, cmap='gray')
    ax.axis('off')

    Ni, Nj = positive_patches[0].shape
    indices = np.array(indices)

    for i, j in indices[labels == 1]:
        ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',alpha=0.3, lw=2, facecolor='none'))
    


